# -*- coding: utf-8 -*-
"""Plotting routines split by diagnostic family."""

from scripts.plot_figs import core as _core
from scipy import stats

# Reuse shared plotting namespace and helpers from core module.
globals().update({k: v for k, v in _core.__dict__.items() if k not in globals()})


def _axes_span_in_figure(axes):
    axes_list = [ax for ax in np.ravel(np.asarray(axes)).tolist() if ax is not None]
    if not axes_list:
        return 0.10, 0.90, 0.10
    boxes = [ax.get_position() for ax in axes_list]
    x0 = float(min(bb.x0 for bb in boxes))
    x1 = float(max(bb.x1 for bb in boxes))
    y0 = float(min(bb.y0 for bb in boxes))
    return x0, x1, y0


def _legend_ncol_by_width(fig, labels, width_frac, target_rows, fontsize):
    n_items = max(1, len(labels))
    fs = float(fontsize)
    fig_width_px = max(1.0, float(fig.get_figwidth()) * float(fig.dpi) * max(0.10, float(width_frac)))
    widest_chars = max(6, max((len(str(lb)) for lb in labels), default=6))
    char_px = fs * float(fig.dpi) / 72.0 * 0.60
    item_px = max(70.0, char_px * (widest_chars + 8.0))
    max_fit = max(1, int(fig_width_px // item_px))
    by_rows = max(1, int(math.ceil(n_items / max(1, int(target_rows)))))
    return max(1, min(n_items, max_fit, by_rows))


def _place_adaptive_bottom_legend(fig, axes, handles, labels, *,
                                  target_rows=3, fontsize=None, y_gap=0.020):
    if not handles or not labels:
        return None
    fs = _FS_LEGEND if fontsize is None else float(fontsize)
    x0, x1, _ = _axes_span_in_figure(axes)
    width = max(0.15, x1 - x0)
    ncol = _legend_ncol_by_width(fig, labels, width, target_rows, fs)
    n_rows = int(math.ceil(len(labels) / float(max(1, ncol))))
    bottom_margin = min(
        0.46,
        max(0.14, 0.07 + 0.056 * n_rows) + max(0.0, float(_LINE_LEGEND_BOTTOM_MARGIN_EXTRA)),
    )
    fig.subplots_adjust(bottom=bottom_margin)
    x0, x1, y0 = _axes_span_in_figure(axes)
    width = max(0.15, x1 - x0)
    gap_auto = float(y_gap) + 0.004 * max(0, n_rows - 1)
    legend_top = max(
        0.01,
        y0 - gap_auto - max(0.0, float(_LINE_LEGEND_Y_GAP_EXTRA)),
    )
    return fig.legend(
        handles, labels,
        loc='upper left',
        bbox_to_anchor=(x0, legend_top, width, 0.01),
        bbox_transform=fig.transFigure,
        ncol=ncol,
        mode='expand',
        fontsize=fs,
        frameon=_LINE_LEGEND_FRAMEON,
        borderaxespad=0.0,
    )


def _clear_split_outputs(fig_name: Optional[str]) -> None:
    if not fig_name:
        return
    p = Path(fig_name)
    for old in p.parent.glob(f'{p.stem}_part*{p.suffix}'):
        try:
            old.unlink()
        except Exception:
            pass


def _split_output_name(fig_name: Optional[str], part_idx: int) -> Optional[str]:
    if not fig_name:
        return fig_name
    p = Path(fig_name)
    if int(part_idx) <= 0:
        return str(p)
    return str(p.with_name(f'{p.stem}_part{int(part_idx) + 1}{p.suffix}'))


def _chunk_items(items, labels, chunk_size: int):
    """Return one full chunk so SItrans always renders single-figure outputs."""
    arr = list(items or [])
    lbl = list(labels or [])
    yield 0, arr, lbl[:len(arr)]


def _row_label_text(label: str) -> str:
    txt = str(label)
    return txt.replace('+', '\n+\n')


def _sitrans_cycle_start(hms: str) -> datetime.date:
    """Return a reference cycle-start date for SItrans tick formatting."""
    hms = (hms or '').lower()
    if hms == 'nh':
        # Arctic transition year: Sep -> Aug
        return datetime.date(2001, 9, 1)
    # Antarctic transition year: Mar -> Feb
    return datetime.date(2001, 3, 1)

def _sitrans_day_to_label(day_val, hms: str) -> str:
    """Convert SItrans day index (0-based) to MM-DD label."""
    if not np.isfinite(day_val):
        return ''

    base = _sitrans_cycle_start(hms)
    day_idx = int(round(day_val))
    # Clamp for robust colorbar label rendering.
    day_idx = max(0, min(day_idx, 365))
    label_date = base + datetime.timedelta(days=day_idx)
    return label_date.strftime('%m-%d')

def _sitrans_day_to_month_label(day_val, hms: str) -> str:
    """Convert SItrans day index (0-based) to short month label (e.g., Mar.)."""
    if not np.isfinite(day_val):
        return ''
    base = _sitrans_cycle_start(hms)
    day_idx = int(round(day_val))
    day_idx = max(0, min(day_idx, 365))
    label_date = base + datetime.timedelta(days=day_idx)
    month_map = {
        1: 'Jan.', 2: 'Feb.', 3: 'Mar.', 4: 'Apr.',
        5: 'May.', 6: 'Jun.', 7: 'Jul.', 8: 'Aug.',
        9: 'Sep.', 10: 'Oct.', 11: 'Nov.', 12: 'Dec.',
    }
    return month_map.get(int(label_date.month), label_date.strftime('%b.'))

def _sitrans_colorbar_ticks(hms: str) -> np.ndarray:
    """Return monthly SItrans tick positions in cycle-day coordinates."""
    base = _sitrans_cycle_start(hms)
    ticks = []
    for moff in range(0, 12):
        month = ((base.month - 1 + moff) % 12) + 1
        year = base.year + ((base.month - 1 + moff) // 12)
        tick_date = datetime.date(year, month, 1)
        ticks.append((tick_date - base).days)
    ticks.append(365)
    return np.array(sorted(set(ticks)), dtype=float)

def _sitrans_discrete_bounds_and_ticks(hms: str, vmin: float, vmax: float):
    """Build clipped discrete boundaries and centered tick positions for SItrans maps."""
    month_ticks = _sitrans_colorbar_ticks(hms)
    inner_bounds = month_ticks[(month_ticks > vmin) & (month_ticks < vmax)]
    bounds = np.concatenate(([vmin], inner_bounds, [vmax])).astype(float)
    bounds = np.unique(bounds)
    if bounds.size < 2: 
        bounds = np.array([vmin, vmax], dtype=float)
    tick_pos = 0.5 * (bounds[:-1] + bounds[1:])
    tick_labels = [_sitrans_day_to_label(x, hms) for x in tick_pos]
    return bounds, tick_pos, tick_labels

def _sitrans_panel_map(grid_file, field_list, panel_labels, hms, fig_name,
                       vmin, vmax, cmap='viridis', unit='days',
                       value_label_mode='days', use_discrete_dates=False,
                       n_obs: int = 1, plot_mode: str = 'raw',
                       diff_cmap='RdBu_r', sigmask_list=None):
    """Plot one SItrans metric type across all participants in a single master figure."""
    if not field_list:
        return

    with xr.open_dataset(grid_file) as ds:
        lon, lat = ds['lon'], ds['lat']

    fields_all = [np.array(f, dtype=float) for f in field_list]
    labels_all = list(panel_labels or [])
    sig_all = list(sigmask_list or [])
    if len(sig_all) < len(fields_all):
        sig_all.extend([None] * (len(fields_all) - len(sig_all)))
    sig_all = [
        (None if s is None else np.array(s, dtype=float))
        for s in sig_all[:len(fields_all)]
    ]
    total_panels = len(fields_all)
    n_obs = max(1, min(int(n_obs), total_panels))
    max_panels_per_fig = 9

    hms = (hms or '').lower()
    proj = ccrs.Stereographic(central_latitude=-90 if hms == 'sh' else 90, central_longitude=0)
    if use_discrete_dates:
        bounds, ticks, tick_labels = _sitrans_discrete_bounds_and_ticks(hms, vmin, vmax)
        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        norm = matplotlib.colors.BoundaryNorm(bounds, ncolors=cmap_obj.N, clip=True)
    else:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        ticks = tick_labels = None

    _clear_split_outputs(fig_name)
    for part_idx, fields, labels_chunk in _chunk_items(fields_all, labels_all, max_panels_per_fig):
        sig_chunk = sig_all[:len(fields)]
        n_panels = len(fields)
        if n_panels == 0:
            continue
        n_rows, n_cols = _adaptive_grid(n_panels)
        fig, ax = plt.subplots(
            n_rows, n_cols,
            figsize=_map_figsize(n_cols, n_rows),
            subplot_kw={'projection': proj}
        )
        ax = np.atleast_1d(ax).ravel()
        global_offset = part_idx * max_panels_per_fig
        use_diff_mode = str(plot_mode or 'raw').lower() == 'diff' and n_panels >= 2
        ref_field = np.array(fields[0], dtype=float) if use_diff_mode else None
        diff_vmax = None
        if use_diff_mode:
            diff_stack = []
            for ii, field in enumerate(fields[1:], start=1):
                dd = np.array(field, dtype=float) - ref_field
                if (global_offset + ii) >= n_obs:
                    dd = _mask_model_zeros(dd)
                diff_stack.append(np.ravel(dd))
            diff_vals = np.concatenate(diff_stack) if diff_stack else np.array([], dtype=float)
            diff_finite = diff_vals[np.isfinite(diff_vals)]
            if diff_finite.size:
                diff_vmax = float(np.nanpercentile(np.abs(diff_finite), 95))
            else:
                diff_vmax = float(max(abs(vmin), abs(vmax), 1.0))
            min_floor = 0.05 if value_label_mode == 'days/year' else 1.0
            diff_vmax = max(min_floor, diff_vmax)

        im_raw = None
        im_diff = None
        all_raw_values = []
        all_diff_values = []
        for ii, field in enumerate(fields):
            lbl = labels_chunk[ii] if ii < len(labels_chunk) else f'Dataset {global_offset + ii + 1}'
            raw_field = np.array(field, dtype=float)
            if (global_offset + ii) >= n_obs:
                raw_field = _mask_model_zeros(raw_field)

            if use_diff_mode and ii > 0:
                diff_field = raw_field - ref_field
                diff_field = _mask_model_zeros(diff_field)
                im_diff = polar_map(hms, ax[ii]).pcolormesh(
                    lon, lat, diff_field,
                    vmin=-diff_vmax, vmax=diff_vmax,
                    transform=ccrs.PlateCarree(),
                    cmap=diff_cmap,
                )
                all_diff_values.append(np.ravel(diff_field))
            else:
                im_raw = polar_map(hms, ax[ii]).pcolormesh(
                    lon, lat, raw_field,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap,
                    norm=norm,
                )
                all_raw_values.append(np.ravel(raw_field))

            sigmask = sig_chunk[ii] if ii < len(sig_chunk) else None
            if sigmask is not None:
                _overlay_sig_markers(ax[ii], lon, lat, sigmask)
            ax[ii].set_title(lbl, fontsize=_FS_SUBPLOT_TITLE)

        for jj in range(n_panels, len(ax)):
            fig.delaxes(ax[jj])

        used_axes = ax[:n_panels].tolist()
        _finalize_map_layout(fig, has_bottom_cbar=False)
        cbar_label = 'MM-DD' if value_label_mode == 'day' else ('days/year' if value_label_mode == 'days/year' else unit)
        cbar_tick_fs = _MAP_CBAR_TICK_FONTSIZE if _MAP_CBAR_TICK_FONTSIZE is not None else _FS_CBAR_TICK
        if use_diff_mode and im_diff is not None:
            diff_label = 'Δ days/year' if value_label_mode == 'days/year' else 'Δ days'
            diff_panel_values = np.concatenate(all_diff_values) if all_diff_values else np.array([], dtype=float)
            _bottom_cbar(
                fig, im_diff, used_axes,
                label=diff_label, extend='auto',
                data=diff_panel_values,
            )
        elif im_raw is not None:
            raw_panel_values = np.concatenate(all_raw_values) if all_raw_values else np.array([], dtype=float)
            cbar = _bottom_cbar(
                fig, im_raw, used_axes,
                label=cbar_label, extend='auto',
                data=raw_panel_values,
            )
            if use_discrete_dates and ticks is not None and tick_labels is not None:
                cbar.set_ticks(np.asarray(ticks, dtype=float))
                cbar.ax.set_xticklabels(list(tick_labels), fontsize=cbar_tick_fs)
                cbar.ax.tick_params(labelsize=cbar_tick_fs)

        _save_fig(_split_output_name(fig_name, part_idx))

def advance_climatology_map(grid_file, field_list, panel_labels, hms, fig_name,
                            cmap='viridis', n_obs: int = 1,
                            plot_mode: str = 'raw', diff_cmap='RdBu_r'):
    """Master figure for SItrans advance climatology (Obs1/Obs2/models)."""
    hms = (hms or '').lower()
    vmin, vmax = (0, 240) if hms == 'nh' else (60, 240)
    _sitrans_panel_map(
        grid_file, field_list, panel_labels, hms, fig_name,
        vmin=vmin, vmax=vmax, cmap=plt.get_cmap(cmap, 12),
        unit='days', value_label_mode='day', use_discrete_dates=True, n_obs=n_obs,
        plot_mode=plot_mode, diff_cmap=diff_cmap,
    )

def retreat_climatology_map(grid_file, field_list, panel_labels, hms, fig_name,
                            cmap='viridis', n_obs: int = 1,
                            plot_mode: str = 'raw', diff_cmap='RdBu_r'):
    """Master figure for SItrans retreat climatology (Obs1/Obs2/models)."""
    _sitrans_panel_map(
        grid_file, field_list, panel_labels, hms, fig_name,
        vmin=120, vmax=365, cmap=plt.get_cmap(cmap, 12),
        unit='days', value_label_mode='day', use_discrete_dates=True, n_obs=n_obs,
        plot_mode=plot_mode, diff_cmap=diff_cmap,
    )

def advance_std_map(grid_file, field_list, panel_labels, hms, fig_name,
                    cmap='Purples', n_obs: int = 1,
                    plot_mode: str = 'raw', diff_cmap='RdBu_r'):
    """Master figure for SItrans advance variability (std)."""
    vals = np.concatenate([np.ravel(np.array(f, dtype=float)) for f in field_list])
    finite = vals[np.isfinite(vals)]
    vmax = np.nanpercentile(finite, 95) if finite.size else 30.0
    vmax = max(1.0, float(vmax))
    _sitrans_panel_map(
        grid_file, field_list, panel_labels, hms, fig_name,
        vmin=0.0, vmax=vmax, cmap=cmap,
        unit='days', value_label_mode='days', use_discrete_dates=False, n_obs=n_obs,
        plot_mode=plot_mode, diff_cmap=diff_cmap,
    )

def retreat_std_map(grid_file, field_list, panel_labels, hms, fig_name,
                    cmap='Purples', n_obs: int = 1,
                    plot_mode: str = 'raw', diff_cmap='RdBu_r'):
    """Master figure for SItrans retreat variability (std)."""
    vals = np.concatenate([np.ravel(np.array(f, dtype=float)) for f in field_list])
    finite = vals[np.isfinite(vals)]
    vmax = np.nanpercentile(finite, 95) if finite.size else 30.0
    vmax = max(1.0, float(vmax))
    _sitrans_panel_map(
        grid_file, field_list, panel_labels, hms, fig_name,
        vmin=0.0, vmax=vmax, cmap=cmap,
        unit='days', value_label_mode='days', use_discrete_dates=False, n_obs=n_obs,
        plot_mode=plot_mode, diff_cmap=diff_cmap,
    )

def advance_trend_map(grid_file, field_list, panel_labels, hms, fig_name,
                      cmap='RdBu_r', n_obs: int = 1,
                      plot_mode: str = 'raw', sigmask_list=None):
    """Master figure for SItrans advance trend (days/year)."""
    vals = np.concatenate([np.ravel(np.array(f, dtype=float)) for f in field_list])
    finite = vals[np.isfinite(vals)]
    vmax = np.nanpercentile(np.abs(finite), 95) if finite.size else 0.5
    vmax = max(0.05, float(vmax))
    _sitrans_panel_map(
        grid_file, field_list, panel_labels, hms, fig_name,
        vmin=-vmax, vmax=vmax, cmap=cmap,
        unit='days/year', value_label_mode='days/year', use_discrete_dates=False, n_obs=n_obs,
        plot_mode=plot_mode, diff_cmap=cmap, sigmask_list=sigmask_list,
    )

def retreat_trend_map(grid_file, field_list, panel_labels, hms, fig_name,
                      cmap='RdBu_r', n_obs: int = 1,
                      plot_mode: str = 'raw', sigmask_list=None):
    """Master figure for SItrans retreat trend (days/year)."""
    vals = np.concatenate([np.ravel(np.array(f, dtype=float)) for f in field_list])
    finite = vals[np.isfinite(vals)]
    vmax = np.nanpercentile(np.abs(finite), 95) if finite.size else 0.5
    vmax = max(0.05, float(vmax))
    _sitrans_panel_map(
        grid_file, field_list, panel_labels, hms, fig_name,
        vmin=-vmax, vmax=vmax, cmap=cmap,
        unit='days/year', value_label_mode='days/year', use_discrete_dates=False, n_obs=n_obs,
        plot_mode=plot_mode, diff_cmap=cmap, sigmask_list=sigmask_list,
    )

def plot_sitrans_iiee_all(curves_list, panel_labels, fig_name='', hms: str = 'nh'):
    """Plot consolidated SItrans IIEE curves for obs2 baseline and all models."""
    if not curves_list:
        return

    curves_all = list(curves_list)
    labels_all = list(panel_labels or [])
    max_panels_per_fig = 9

    ymax = 0.0
    for c in curves_all:
        aO = np.array(c.get('advance_over', []), dtype=float)
        aU = np.array(c.get('advance_under', []), dtype=float)
        rO = np.array(c.get('retreat_over', []), dtype=float)
        rU = np.array(c.get('retreat_under', []), dtype=float)
        for arr in (aO, aU, rO, rU):
            if arr.size:
                ymax = max(ymax, float(np.nanmax(np.abs(arr))))
        if aO.size and aU.size:
            ymax = max(ymax, float(np.nanmax(np.abs(aO) + np.abs(aU))))
        if rO.size and rU.size:
            ymax = max(ymax, float(np.nanmax(np.abs(rO) + np.abs(rU))))
    ymax = max(0.5, ymax * 1.1)

    legend_handles = [
        plt.Line2D([0], [0], color='m', linestyle='-', marker='.', linewidth=_line_model_width(1.0), label='Advance over'),
        plt.Line2D([0], [0], color='m', linestyle='--', linewidth=_line_model_width(1.0), label='Advance under'),
        plt.Line2D([0], [0], color='r', linestyle='-', linewidth=_line_model_width(1.8), label='Advance IIEE'),
        plt.Line2D([0], [0], color='c', linestyle='-', marker='.', linewidth=_line_model_width(1.0), label='Retreat over'),
        plt.Line2D([0], [0], color='c', linestyle='--', linewidth=_line_model_width(1.0), label='Retreat under'),
        plt.Line2D([0], [0], color='b', linestyle='-', linewidth=_line_model_width(1.8), label='Retreat IIEE'),
    ]
    legend_labels = [h.get_label() for h in legend_handles]

    def _compact_title(label_text: str) -> str:
        txt = str(label_text or '').strip()
        low = txt.lower()
        cut = low.find(' vs ')
        if cut >= 0:
            txt = txt[:cut].strip()
        return txt or str(label_text)

    hms = (hms or '').lower()
    month_ticks_all = _sitrans_colorbar_ticks(hms)
    try:
        month_tick_rotation = float(_LINE_MONTH_TICK_ROTATION)
    except Exception:
        month_tick_rotation = 25.0
    month_tick_rotation = min(45.0, max(15.0, abs(month_tick_rotation)))
    _clear_split_outputs(fig_name)
    for part_idx, curves_chunk, labels_chunk in _chunk_items(curves_all, labels_all, max_panels_per_fig):
        n_panels = len(curves_chunk)
        if n_panels == 0:
            continue
        n_rows, n_cols = _adaptive_grid(n_panels)
        fig, ax = plt.subplots(
            n_rows, n_cols,
            figsize=_line_figsize(max(10.8, 5.2 * n_cols), max(4.2, 3.9 * n_rows)),
            sharex=True, sharey=True,
        )
        ax = np.atleast_1d(ax).ravel()
        global_offset = part_idx * max_panels_per_fig

        for ii, c in enumerate(curves_chunk):
            days = np.array(c.get('days_range', []), dtype=float)
            aO = np.array(c.get('advance_over', []), dtype=float)
            aU = np.array(c.get('advance_under', []), dtype=float)
            rO = np.array(c.get('retreat_over', []), dtype=float)
            rU = np.array(c.get('retreat_under', []), dtype=float)

            ax[ii].plot(days, aO, 'm.-', linewidth=_line_model_width(1.0))
            ax[ii].plot(days, aU, 'm--', linewidth=_line_model_width(1.0))
            ax[ii].plot(days, np.abs(aO) + np.abs(aU), 'r-', linewidth=_line_model_width(1.8))
            ax[ii].plot(days, rO, 'c.-', linewidth=_line_model_width(1.0))
            ax[ii].plot(days, rU, 'c--', linewidth=_line_model_width(1.0))
            ax[ii].plot(days, np.abs(rO) + np.abs(rU), 'b-', linewidth=_line_model_width(1.8))
            ax[ii].axhline(y=0, **_line_zero_style(color='gray', alpha=0.5))

            title = labels_chunk[ii] if ii < len(labels_chunk) else f'Dataset {global_offset + ii + 1}'
            ax[ii].set_title(_compact_title(title), fontsize=_FS_SUBPLOT_TITLE)
            ax[ii].tick_params(axis='both', labelsize=_FS_TICK)
            _apply_light_grid(ax[ii])
            ax[ii].set_ylim(0, ymax)
            row_idx, col_idx = divmod(ii, n_cols)
            finite_days = days[np.isfinite(days)]
            if finite_days.size:
                xmin = float(np.nanmin(finite_days))
                xmax = float(np.nanmax(finite_days))
                ax[ii].set_xlim(xmin, xmax)
                ticks = month_ticks_all[(month_ticks_all >= xmin - 2.0) & (month_ticks_all <= xmax + 2.0)]
                if ticks.size < 3:
                    ticks = np.linspace(xmin, xmax, 6)
                if ticks.size:
                    ax[ii].set_xticks(ticks)
                    labels_month = [_sitrans_day_to_month_label(t, hms) for t in ticks]
                    labels_sparse = [lb if (kk % 2 == 0) else '' for kk, lb in enumerate(labels_month)]
                    ax[ii].set_xticklabels(
                        labels_sparse,
                        fontsize=max(12, _FS_TICK + 2),
                        rotation=month_tick_rotation,
                        ha='right',
                    )
                    ax[ii].tick_params(axis='x', which='major', length=4.5, width=0.9)
            ax[ii].tick_params(axis='x', labelbottom=True)
            ax[ii].tick_params(axis='y', labelleft=True)
            ax[ii].set_xlabel('Month', fontsize=_FS_AXIS_LABEL)
            ax[ii].set_ylabel('Area Difference (10$^6$ km$^2$)', fontsize=_FS_AXIS_LABEL)

        for jj in range(n_panels, len(ax)):
            fig.delaxes(ax[jj])

        _finalize_line_layout(fig, legend_location='none', pad=0.80)
        raw_wspace = _line_subplot_wspace(0.12)
        raw_hspace = _line_subplot_hspace(0.34)
        legend_ncol = max(1, min(3, len(legend_labels)))
        legend_rows = int(math.ceil(len(legend_labels) / float(max(1, legend_ncol))))
        top_axes = max(0.70, 0.94 - 0.07 * legend_rows)
        legend_y = min(0.995, top_axes + 0.018)
        fig.subplots_adjust(
            left=0.07, right=0.985, top=top_axes, bottom=0.10,
            wspace=min(0.24, max(0.04, float(raw_wspace))),
            hspace=max(0.50, float(raw_hspace)),
        )
        fig.legend(
            legend_handles,
            legend_labels,
            loc='lower left',
            bbox_to_anchor=(0.29875, legend_y, 0.4575, 0.01),
            bbox_transform=fig.transFigure,
            mode='expand',
            ncol=legend_ncol,
            fontsize=max(15, _FS_LEGEND + 5),
            frameon=_LINE_LEGEND_FRAMEON,
            borderaxespad=0.0,
            labelspacing=0.42,
            columnspacing=0.85,
            handletextpad=0.45,
        )
        _save_fig(_split_output_name(fig_name, part_idx))

def _overlay_sig_markers(ax, lon, lat, sigmask: np.ndarray):
    """Overlay sparse significance markers on one map axis."""
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    sig = np.asarray(sigmask, dtype=float)
    sig = np.isfinite(sig) & (sig > 0.5)
    if not np.any(sig):
        return

    nx = sig.shape[0]
    step = max(1, int(nx / 36))
    lon_sub = lon[::step, ::step]
    lat_sub = lat[::step, ::step]
    sig_sub = sig[::step, ::step]
    lon_sig = lon_sub[sig_sub]
    lat_sig = lat_sub[sig_sub]
    if lon_sig.size == 0:
        return
    ax.scatter(
        lon_sig, lat_sig,
        s=12, c='0.35', alpha=0.65,
        transform=ccrs.PlateCarree(),
        marker='x', linewidths=0.45,
    )

def plot_sitrans_bias_map(grid_file, metrics_list, panel_labels, hms, fig_name=None,
                          cmap='RdBu_r', phase: str = 'advance'):
    """Plot SItrans bias maps (model - obs) for one phase (advance/retreat)."""
    with xr.open_dataset(grid_file) as ds:
        lon, lat = ds['lon'], ds['lat']

    metrics_all = list(metrics_list or [])
    labels_all = list(panel_labels or [])
    if not metrics_all:
        return

    phase_norm = str(phase or 'advance').strip().lower()
    if phase_norm.startswith('ret'):
        phase_key = 'retreat_bias_map'
        phase_title = 'Retreat bias'
    else:
        phase_key = 'advance_bias_map'
        phase_title = 'Advance bias'

    fields = []
    labels_used = []
    for ii, metrics in enumerate(metrics_all):
        if phase_key not in metrics:
            continue
        fields.append(np.array(metrics[phase_key], dtype=float))
        labels_used.append(labels_all[ii] if ii < len(labels_all) else f'Model {ii + 1}')
    if not fields:
        return

    all_bias_vals = np.concatenate([np.ravel(np.asarray(f, dtype=float)) for f in fields])
    finite = all_bias_vals[np.isfinite(all_bias_vals)]
    vmax = np.nanpercentile(np.abs(finite), 95) if finite.size else 30.0
    vmax = max(5.0, float(vmax))
    vmin = -vmax

    n_panels = len(fields)
    n_rows, n_cols = _adaptive_grid(n_panels)
    proj = ccrs.Stereographic(central_latitude=-90 if hms == 'sh' else 90, central_longitude=0)
    _clear_split_outputs(fig_name)
    fig, ax = plt.subplots(
        n_rows, n_cols,
        figsize=_map_figsize(n_cols, n_rows),
        subplot_kw={'projection': proj},
    )
    ax = np.atleast_1d(ax).ravel()
    im = None
    for ii, field in enumerate(fields):
        im = polar_map(hms, ax[ii]).pcolormesh(
            lon, lat, field,
            vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
        )
        ax[ii].set_title(labels_used[ii], fontsize=_FS_SUBPLOT_TITLE)
    for jj in range(n_panels, len(ax)):
        fig.delaxes(ax[jj])

    used_axes = ax[:n_panels].tolist()
    _finalize_map_layout(fig, has_bottom_cbar=False)
    _bottom_cbar(
        fig, im, used_axes,
        label='days', extend='auto',
        data=all_bias_vals,
    )
    fig.suptitle(phase_title, fontsize=_FS_MAIN_TITLE, y=0.995)
    _save_fig(fig_name)

def plot_sitrans_std_map(grid_file, metrics_list, panel_labels, hms, fig_name=None,
                         std_cmap='Purples', diff_cmap='RdBu_r'):
    """Plot SItrans variability maps: obs STD, model STD, and STD difference."""
    with xr.open_dataset(grid_file) as ds:
        lon, lat = ds['lon'], ds['lat']

    metrics_all = list(metrics_list or [])
    labels_all = list(panel_labels or [])
    n_models = len(metrics_all)
    if n_models == 0:
        return

    all_std = []
    all_diff = []
    for metrics in metrics_all:
        all_std.extend([
            np.ravel(np.array(metrics['advance_std_obs'], dtype=float)),
            np.ravel(np.array(metrics['advance_std_mod'], dtype=float)),
            np.ravel(np.array(metrics['retreat_std_obs'], dtype=float)),
            np.ravel(np.array(metrics['retreat_std_mod'], dtype=float)),
        ])
        all_diff.extend([
            np.ravel(np.array(metrics['advance_std_diff_map'], dtype=float)),
            np.ravel(np.array(metrics['retreat_std_diff_map'], dtype=float)),
        ])

    std_vals = np.concatenate(all_std)
    std_finite = std_vals[np.isfinite(std_vals)]
    std_vmax = np.nanpercentile(std_finite, 95) if std_finite.size else 30.0
    std_vmax = max(1.0, float(std_vmax))

    diff_vals = np.concatenate(all_diff)
    diff_finite = diff_vals[np.isfinite(diff_vals)]
    diff_vmax = np.nanpercentile(np.abs(diff_finite), 95) if diff_finite.size else 10.0
    diff_vmax = max(1.0, float(diff_vmax))

    titles = [
        'Advance STD obs', 'Advance STD model', 'Advance STD diff',
        'Retreat STD obs', 'Retreat STD model', 'Retreat STD diff',
    ]
    max_rows_per_fig = 4
    proj = ccrs.Stereographic(central_latitude=-90 if hms == 'sh' else 90, central_longitude=0)
    _clear_split_outputs(fig_name)
    for part_idx, metrics_chunk, labels_chunk in _chunk_items(metrics_all, labels_all, max_rows_per_fig):
        rows = len(metrics_chunk)
        if rows == 0:
            continue
        fig, ax = plt.subplots(
            rows, 6,
            figsize=_map_abs_figsize(6 * 2.95, max(6.8, rows * 3.35)),
            subplot_kw={'projection': proj},
        )
        if rows == 1:
            ax = np.array([ax])

        for ii, metrics in enumerate(metrics_chunk):
            fields = [
                np.array(metrics['advance_std_obs'], dtype=float),
                np.array(metrics['advance_std_mod'], dtype=float),
                np.array(metrics['advance_std_diff_map'], dtype=float),
                np.array(metrics['retreat_std_obs'], dtype=float),
                np.array(metrics['retreat_std_mod'], dtype=float),
                np.array(metrics['retreat_std_diff_map'], dtype=float),
            ]
            for jj, field in enumerate(fields):
                if jj in (1, 4):
                    field = _mask_model_zeros(field)
                if jj in (2, 5):
                    polar_map(hms, ax[ii, jj]).pcolormesh(
                        lon, lat, field, vmin=-diff_vmax, vmax=diff_vmax,
                        transform=ccrs.PlateCarree(), cmap=diff_cmap)
                else:
                    polar_map(hms, ax[ii, jj]).pcolormesh(
                        lon, lat, field, vmin=0, vmax=std_vmax,
                        transform=ccrs.PlateCarree(), cmap=std_cmap)
                if ii == 0:
                    ax[ii, jj].set_title(titles[jj], fontsize=_FS_SUBPLOT_TITLE, fontweight='bold')
                if jj == 0:
                    row_label = labels_chunk[ii] if ii < len(labels_chunk) else f'Model {part_idx * max_rows_per_fig + ii + 1}'
                    row_label = _row_label_text(row_label)
                    ax[ii, jj].text(
                        -0.23, 0.5, row_label,
                        transform=ax[ii, jj].transAxes,
                        fontsize=max(12, _FS_SUBPLOT_TITLE - 3),
                        fontweight='bold',
                        rotation=90,
                        va='center',
                        ha='center',
                        linespacing=1.05,
                    )

        sm_std = plt.cm.ScalarMappable(cmap=std_cmap, norm=plt.Normalize(vmin=0, vmax=std_vmax))
        sm_std.set_array([])
        sm_diff = plt.cm.ScalarMappable(cmap=diff_cmap, norm=plt.Normalize(vmin=-diff_vmax, vmax=diff_vmax))
        sm_diff.set_array([])

        _finalize_map_layout(fig, has_bottom_cbar=False, pad=0.72)
        _bottom_cbar(
            fig, sm_std, ax.ravel().tolist(),
            label='STD (days)', extend='auto',
            data=std_finite,
        )
        _bottom_cbar(
            fig, sm_diff, ax.ravel().tolist(),
            label='Difference (days)', extend='auto',
            pad=0.115,
            data=diff_finite,
        )
        _save_fig(_split_output_name(fig_name, part_idx))

def plot_sitrans_trend_map(grid_file, metrics_list, panel_labels, hms, fig_name=None, cmap='RdBu_r'):
    """Plot SItrans trend maps: obs trend, model trend, and trend difference."""
    with xr.open_dataset(grid_file) as ds:
        lon, lat = ds['lon'], ds['lat']

    metrics_all = list(metrics_list or [])
    labels_all = list(panel_labels or [])
    n_models = len(metrics_all)
    if n_models == 0:
        return

    all_trend = []
    for metrics in metrics_all:
        all_trend.extend([
            np.ravel(np.array(metrics['advance_trend_obs'], dtype=float)),
            np.ravel(np.array(metrics['advance_trend_mod'], dtype=float)),
            np.ravel(np.array(metrics['advance_trend_diff_map'], dtype=float)),
            np.ravel(np.array(metrics['retreat_trend_obs'], dtype=float)),
            np.ravel(np.array(metrics['retreat_trend_mod'], dtype=float)),
            np.ravel(np.array(metrics['retreat_trend_diff_map'], dtype=float)),
        ])
    all_trend = np.concatenate(all_trend)
    finite = all_trend[np.isfinite(all_trend)]
    vmax = np.nanpercentile(np.abs(finite), 95) if finite.size else 0.5
    vmax = max(0.05, float(vmax))

    titles = [
        'Advance trend obs', 'Advance trend model', 'Advance trend diff',
        'Retreat trend obs', 'Retreat trend model', 'Retreat trend diff',
    ]
    keys = [
        'advance_trend_obs', 'advance_trend_mod', 'advance_trend_diff_map',
        'retreat_trend_obs', 'retreat_trend_mod', 'retreat_trend_diff_map',
    ]

    max_rows_per_fig = 4
    proj = ccrs.Stereographic(central_latitude=-90 if hms == 'sh' else 90, central_longitude=0)
    _clear_split_outputs(fig_name)
    for part_idx, metrics_chunk, labels_chunk in _chunk_items(metrics_all, labels_all, max_rows_per_fig):
        rows = len(metrics_chunk)
        if rows == 0:
            continue
        fig, ax = plt.subplots(
            rows, 6,
            figsize=_map_abs_figsize(6 * 2.95, max(6.8, rows * 3.35)),
            subplot_kw={'projection': proj},
        )
        if rows == 1:
            ax = np.array([ax])
        im = None
        for ii, metrics in enumerate(metrics_chunk):
            for jj, key in enumerate(keys):
                field = np.array(metrics[key], dtype=float)
                if key in ('advance_trend_mod', 'retreat_trend_mod'):
                    field = _mask_model_zeros(field)
                im = polar_map(hms, ax[ii, jj]).pcolormesh(
                    lon, lat, field, vmin=-vmax, vmax=vmax,
                    transform=ccrs.PlateCarree(), cmap=cmap)
                if ii == 0:
                    ax[ii, jj].set_title(titles[jj], fontsize=_FS_SUBPLOT_TITLE, fontweight='bold')
                if jj == 0:
                    row_label = labels_chunk[ii] if ii < len(labels_chunk) else f'Model {part_idx * max_rows_per_fig + ii + 1}'
                    row_label = _row_label_text(row_label)
                    ax[ii, jj].text(
                        -0.23, 0.5, row_label,
                        transform=ax[ii, jj].transAxes,
                        fontsize=max(12, _FS_SUBPLOT_TITLE - 3),
                        fontweight='bold',
                        rotation=90,
                        va='center',
                        ha='center',
                        linespacing=1.05,
                    )

                sig_key = {
                    'advance_trend_obs': 'advance_trend_sigmask_obs',
                    'advance_trend_mod': 'advance_trend_sigmask_mod',
                    'retreat_trend_obs': 'retreat_trend_sigmask_obs',
                    'retreat_trend_mod': 'retreat_trend_sigmask_mod',
                }.get(key)
                if sig_key and sig_key in metrics:
                    _overlay_sig_markers(
                        ax[ii, jj],
                        lon, lat,
                        np.array(metrics[sig_key], dtype=float),
                    )

        _finalize_map_layout(fig, has_bottom_cbar=False, pad=0.72)
        _bottom_cbar(
            fig, im, ax.ravel().tolist(),
            label='days/year', extend='auto',
            data=all_trend,
        )

        _save_fig(_split_output_name(fig_name, part_idx))

def plot_sitrans_significance_map(grid_file, metrics_list, panel_labels, hms, fig_name=None):
    """Plot SItrans trend-significance masks (model trend p < threshold)."""
    with xr.open_dataset(grid_file) as ds:
        lon, lat = ds['lon'], ds['lat']

    metrics_all = list(metrics_list or [])
    labels_all = list(panel_labels or [])
    n_models = len(metrics_all)
    if n_models == 0:
        return

    max_rows_per_fig = 6
    proj = ccrs.Stereographic(central_latitude=-90 if hms == 'sh' else 90, central_longitude=0)
    cmp = plt.get_cmap('Greys', 4)
    _clear_split_outputs(fig_name)
    for part_idx, metrics_chunk, labels_chunk in _chunk_items(metrics_all, labels_all, max_rows_per_fig):
        rows = len(metrics_chunk)
        if rows == 0:
            continue
        fig, ax = plt.subplots(
            rows, 2,
            figsize=_map_abs_figsize(8.0, max(6.2, rows * 3.35)),
            subplot_kw={'projection': proj},
        )
        if rows == 1:
            ax = np.array([ax])

        im = None
        for ii, metrics in enumerate(metrics_chunk):
            adv_sig = np.array(metrics.get('advance_trend_sigmask_mod', np.full(lon.shape, np.nan)), dtype=float)
            ret_sig = np.array(metrics.get('retreat_trend_sigmask_mod', np.full(lon.shape, np.nan)), dtype=float)

            im = polar_map(hms, ax[ii, 0]).pcolormesh(
                lon, lat, adv_sig, vmin=0.0, vmax=1.0,
                transform=ccrs.PlateCarree(), cmap=cmp)
            polar_map(hms, ax[ii, 1]).pcolormesh(
                lon, lat, ret_sig, vmin=0.0, vmax=1.0,
                transform=ccrs.PlateCarree(), cmap=cmp)

            if ii == 0:
                ax[ii, 0].set_title('Advance trend significant', fontsize=_FS_SUBPLOT_TITLE, fontweight='bold')
                ax[ii, 1].set_title('Retreat trend significant', fontsize=_FS_SUBPLOT_TITLE, fontweight='bold')
            row_label = labels_chunk[ii] if ii < len(labels_chunk) else f'Model {part_idx * max_rows_per_fig + ii + 1}'
            row_label = _row_label_text(row_label)
            ax[ii, 0].text(
                -0.23, 0.5, row_label,
                transform=ax[ii, 0].transAxes,
                fontsize=max(12, _FS_SUBPLOT_TITLE - 3),
                fontweight='bold',
                rotation=90,
                va='center',
                ha='center',
                linespacing=1.05,
            )

        _finalize_map_layout(fig, has_bottom_cbar=False, pad=0.72)
        if im is not None:
            _bottom_cbar(
                fig, im, ax.ravel().tolist(),
                label='1 = significant', extend='neither',
            )
        _save_fig(_split_output_name(fig_name, part_idx))

def plot_sitrans_relationship_scatter(rel_list, panel_labels, fig_name='',
                                      obs_ref_label: str = 'Obs1',
                                      comparator_label: str = 'Comparator'):
    """Plot retreat(t) vs advance(t+1) scatter for one reference and comparators."""
    if not rel_list:
        return

    payload_all = list(rel_list)
    labels_all = list(panel_labels or [])
    max_panels_per_fig = 9
    _clear_split_outputs(fig_name)
    for part_idx, payload_chunk, labels_chunk in _chunk_items(payload_all, labels_all, max_panels_per_fig):
        n_panels = len(payload_chunk)
        if n_panels == 0:
            continue
        n_rows, n_cols = _adaptive_grid(n_panels)
        fig, ax = plt.subplots(
            n_rows, n_cols,
            figsize=_line_figsize(max(10.0, 4.2 * n_cols), max(4.0, 3.6 * n_rows)),
            sharex=True, sharey=True,
        )
        ax = np.atleast_1d(ax).ravel()
        global_offset = part_idx * max_panels_per_fig

        for ii, payload in enumerate(payload_chunk):
            obs_x = np.asarray(payload.get('obs_retreat_t', []), dtype=float)
            obs_y = np.asarray(payload.get('obs_advance_t1', []), dtype=float)
            mod_x = np.asarray(payload.get('mod_retreat_t', []), dtype=float)
            mod_y = np.asarray(payload.get('mod_advance_t1', []), dtype=float)

            valid_obs = np.isfinite(obs_x) & np.isfinite(obs_y)
            valid_mod = np.isfinite(mod_x) & np.isfinite(mod_y)

            if np.any(valid_obs):
                ax[ii].scatter(
                    obs_x[valid_obs], obs_y[valid_obs],
                    c='k', s=22, alpha=0.75, label=str(obs_ref_label),
                )
                if np.sum(valid_obs) >= 2:
                    reg_obs = stats.linregress(obs_x[valid_obs], obs_y[valid_obs])
                    xfit = np.linspace(np.nanmin(obs_x[valid_obs]), np.nanmax(obs_x[valid_obs]), 100)
                    ax[ii].plot(xfit, reg_obs.intercept + reg_obs.slope * xfit, 'k--', lw=1.0, alpha=0.85)

            if np.any(valid_mod):
                ax[ii].scatter(
                    mod_x[valid_mod], mod_y[valid_mod],
                    c='tab:red', s=22, alpha=0.70, label=str(comparator_label),
                )
                if np.sum(valid_mod) >= 2:
                    reg_mod = stats.linregress(mod_x[valid_mod], mod_y[valid_mod])
                    xfit = np.linspace(np.nanmin(mod_x[valid_mod]), np.nanmax(mod_x[valid_mod]), 100)
                    ax[ii].plot(xfit, reg_mod.intercept + reg_mod.slope * xfit, color='tab:red', lw=1.0, alpha=0.85)

            title = labels_chunk[ii] if ii < len(labels_chunk) else f'Dataset {global_offset + ii + 1}'
            ax[ii].set_title(title, fontsize=_FS_SUBPLOT_TITLE)
            ax[ii].tick_params(axis='both', labelsize=_FS_TICK)
            _apply_light_grid(ax[ii])
            row_idx, col_idx = divmod(ii, n_cols)
            if row_idx == n_rows - 1:
                ax[ii].set_xlabel('Retreat date (t, day index)', fontsize=_FS_AXIS_LABEL)
            if col_idx == 0:
                ax[ii].set_ylabel('Advance date (t+1, day index)', fontsize=_FS_AXIS_LABEL)

        for jj in range(n_panels, len(ax)):
            fig.delaxes(ax[jj])

        handles = []
        labels_legend = []
        if n_panels > 0:
            handles, labels_legend = ax[0].get_legend_handles_labels()
        _finalize_line_layout(fig, legend_location='none', pad=0.84)
        _place_adaptive_bottom_legend(
            fig,
            ax[:n_panels],
            handles,
            labels_legend,
            target_rows=1,
            fontsize=max(10, _FS_LEGEND),
            y_gap=0.036,
        )
        _save_fig(_split_output_name(fig_name, part_idx))

def plot_sitrans_region_timeseries(region_payloads, phase: str, fig_name='', hms: str = 'nh'):
    """Plot regional annual mean transition-date time series."""
    if not region_payloads:
        return

    payload_all = list(region_payloads)
    max_panels_per_fig = 9
    phase_key = 'advance' if str(phase).lower().startswith('adv') else 'retreat'
    hms = (hms or '').lower()
    month_ticks_all = _sitrans_colorbar_ticks(hms)
    _clear_split_outputs(fig_name)
    for part_idx, payload_chunk, _ in _chunk_items(payload_all, [], max_panels_per_fig):
        n_panels = len(payload_chunk)
        if n_panels == 0:
            continue
        n_rows, n_cols = _adaptive_grid(n_panels)
        fig, ax = plt.subplots(
            n_rows, n_cols,
            figsize=_line_figsize(max(10.0, 4.4 * n_cols), max(5.2, 4.2 * n_rows)),
            sharex=False, sharey=False,
        )
        ax = np.atleast_1d(ax).ravel()

        for ii, payload in enumerate(payload_chunk):
            years = np.asarray(payload.get('years', []), dtype=float)
            obs1 = np.asarray(payload.get(f'{phase_key}_obs1', []), dtype=float)
            obs2 = np.asarray(payload.get(f'{phase_key}_obs2', []), dtype=float)
            obs1_label = str(payload.get('obs1_label', 'Obs1'))
            obs2_label = str(payload.get('obs2_label', 'Obs2'))
            model_series = payload.get(f'{phase_key}_models', []) or []
            model_spreads = payload.get(f'{phase_key}_model_spreads', []) or []
            model_labels = payload.get('model_labels', []) or []
            region_name = str(payload.get('region', f'Region {part_idx * max_panels_per_fig + ii + 1}'))
            has_obs2 = bool(years.size and obs2.size == years.size and np.any(np.isfinite(obs2)))
            n_obs = 2 if has_obs2 else 1
            group_rank: Dict[str, int] = {}
            for lb in model_labels:
                if _is_group_label(lb) and lb not in group_rank:
                    group_rank[lb] = len(group_rank)

            if years.size and obs1.size == years.size:
                ax[ii].plot(years, obs1, label=obs1_label, **_obs_style(0))
            if has_obs2:
                ax[ii].plot(years, obs2, label=obs2_label, **_obs_style(1))

            for jj, series in enumerate(model_series):
                yv = np.asarray(series, dtype=float)
                if yv.size != years.size:
                    continue
                label = model_labels[jj] if jj < len(model_labels) else f'model{jj + 1}'
                if _is_group_label(label):
                    style = _group_style(group_rank.get(label, 0))
                else:
                    style_idx = n_obs + jj
                    ls, cr = _get_style(style_idx, n_obs, None, None)
                    style = {
                        'color': cr,
                        'linestyle': ls,
                        'linewidth': _line_model_width(0.85),
                    }
                    alpha = _line_model_alpha(0.85)
                    if alpha is not None:
                        style['alpha'] = alpha
                ax[ii].plot(
                    years, yv,
                    label=label,
                    **style,
                )
                if jj < len(model_spreads):
                    sv = np.asarray(model_spreads[jj], dtype=float)
                    if sv.size == yv.size and np.any(np.isfinite(sv)):
                        _plot_group_std_band(
                            ax[ii],
                            years,
                            yv,
                            sv,
                            style,
                        )

            ax[ii].set_title(region_name, fontsize=max(12, _FS_SUBPLOT_TITLE - 3))
            ax[ii].tick_params(axis='both', labelsize=_FS_TICK)
            _apply_light_grid(ax[ii])
            y0, y1 = ax[ii].get_ylim()
            ylo, yhi = (float(min(y0, y1)), float(max(y0, y1)))
            month_ticks = month_ticks_all[(month_ticks_all >= ylo - 2.0) & (month_ticks_all <= yhi + 2.0)]
            if month_ticks.size < 3 and np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
                month_ticks = np.linspace(ylo, yhi, 5)
            if month_ticks.size:
                ax[ii].set_yticks(month_ticks)
                ax[ii].set_yticklabels(
                    [_sitrans_day_to_month_label(tk, hms) for tk in month_ticks],
                    fontsize=_FS_TICK,
                )
            finite_years = years[np.isfinite(years)]
            if finite_years.size:
                x_min = float(np.nanmin(finite_years))
                x_max = float(np.nanmax(finite_years))
                if np.isfinite(x_min) and np.isfinite(x_max):
                    if x_max > x_min:
                        tick_span = max(1, int(round(x_max - x_min)))
                        tick_step = max(1, int(math.ceil(tick_span / 5.0)))
                        start_tick = int(math.floor(x_min))
                        end_tick = int(math.ceil(x_max))
                        xticks = np.arange(start_tick, end_tick + 1, tick_step, dtype=float)
                        if xticks.size == 0 or xticks[-1] < x_max:
                            xticks = np.append(xticks, float(end_tick))
                        ax[ii].set_xticks(xticks)
                        ax[ii].set_xlim(x_min - 0.4, x_max + 0.4)
                    else:
                        ax[ii].set_xticks([x_min])
                        ax[ii].set_xlim(x_min - 0.5, x_min + 0.5)
            ax[ii].tick_params(axis='x', labelbottom=True, labelsize=_FS_TICK)
            ax[ii].set_xlabel('Transition-year start', fontsize=_FS_AXIS_LABEL)
            _, col_idx = divmod(ii, n_cols)
            if col_idx == 0:
                ax[ii].set_ylabel(f'{phase_key.capitalize()} date (month)', fontsize=_FS_AXIS_LABEL)

        for jj in range(n_panels, len(ax)):
            fig.delaxes(ax[jj])

        handles = []
        labels_legend = []
        if n_panels > 0:
            handles, labels_legend = ax[0].get_legend_handles_labels()
        _finalize_line_layout(fig, legend_location='none', pad=0.84)
        fig.subplots_adjust(hspace=max(0.60, _line_subplot_hspace(0.40)))
        _place_adaptive_bottom_legend(
            fig,
            ax[:n_panels],
            handles,
            labels_legend,
            target_rows=3,
            fontsize=max(9, _FS_LEGEND - 1),
            y_gap=0.052,
        )
        _save_fig(_split_output_name(fig_name, part_idx))

def plot_sitrans_skill_heatmap(matrix: np.ndarray, row_labels, col_labels,
                               fig_name='', cmap='RdBu_r'):
    """Plot model-skill summary heatmap for SItrans diagnostics."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return

    finite = arr[np.isfinite(arr)]
    vmax = np.nanpercentile(np.abs(finite), 95) if finite.size else 1.0
    vmax = max(0.5, float(vmax))

    fig, ax = plt.subplots(
        1, 1,
        figsize=_line_figsize(max(7, 1.2 * arr.shape[1]), max(4, 0.42 * arr.shape[0] + 2.5)),
    )
    im = ax.imshow(arr, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(np.arange(arr.shape[1]))
    ax.set_xticklabels(col_labels, rotation=35, ha='right', fontsize=max(9, _FS_TICK - 2))
    ax.set_yticks(np.arange(arr.shape[0]))
    ax.set_yticklabels(row_labels, fontsize=max(9, _FS_TICK - 2))

    for ii in range(arr.shape[0]):
        for jj in range(arr.shape[1]):
            v = arr[ii, jj]
            if np.isfinite(v):
                ax.text(jj, ii, f'{v:.2f}', ha='center', va='center', fontsize=7, color='black')

    cb = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.045, pad=0.02)
    cb.set_label('Metric value', fontsize=max(10, _FS_CBAR_LABEL - 2))
    cb.ax.tick_params(labelsize=max(8, _FS_CBAR_TICK - 2))
    ax.set_title('SItrans skill summary', fontsize=max(12, _FS_SUBPLOT_TITLE - 2))
    _finalize_layout(fig)
    _save_fig(fig_name)

def plot_sitrans_model_ranking(scores: np.ndarray, labels, fig_name=''):
    """Plot one simple ranking bar chart (smaller score = better)."""
    vals = np.asarray(scores, dtype=float).reshape(-1)
    if vals.size == 0:
        return
    lbls = [str(v) for v in labels]
    if len(lbls) != vals.size:
        lbls = [f'model{i + 1}' for i in range(vals.size)]

    order = np.argsort(np.where(np.isfinite(vals), vals, np.inf))
    vals_sorted = vals[order]
    labels_sorted = [lbls[ii] for ii in order]

    fig, ax = plt.subplots(1, 1, figsize=_line_figsize(8, max(4, 0.45 * vals.size + 1.8)))
    y = np.arange(vals_sorted.size)
    bars = ax.barh(y, vals_sorted, color='tab:blue', alpha=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(labels_sorted, fontsize=max(9, _FS_TICK - 2))
    ax.set_xlabel('Composite error score (lower is better)', fontsize=max(10, _FS_AXIS_LABEL - 2))
    ax.set_title('SItrans model ranking', fontsize=max(12, _FS_SUBPLOT_TITLE - 2))
    _apply_light_grid(ax)

    for ii, bar in enumerate(bars):
        vv = vals_sorted[ii]
        if np.isfinite(vv):
            ax.text(vv, bar.get_y() + bar.get_height() / 2, f' {vv:.2f}',
                    va='center', ha='left', fontsize=8)

    _finalize_layout(fig)
    _save_fig(fig_name)

__all__ = [
    "_sitrans_cycle_start",
    "_sitrans_day_to_label",
    "_sitrans_colorbar_ticks",
    "_sitrans_discrete_bounds_and_ticks",
    "_sitrans_panel_map",
    "advance_climatology_map",
    "retreat_climatology_map",
    "advance_std_map",
    "retreat_std_map",
    "advance_trend_map",
    "retreat_trend_map",
    "plot_sitrans_iiee_all",
    "plot_sitrans_bias_map",
    "plot_sitrans_std_map",
    "plot_sitrans_trend_map",
    "plot_sitrans_significance_map",
    "plot_sitrans_relationship_scatter",
    "plot_sitrans_region_timeseries",
    "plot_sitrans_skill_heatmap",
    "plot_sitrans_model_ranking",
]
