# -*- coding: utf-8 -*-
"""Plotting routines split by diagnostic family."""

from scripts.plot_figs import core as _core

# Reuse shared plotting namespace and helpers from core module.
globals().update({k: v for k, v in _core.__dict__.items() if k not in globals()})


def _axis_span_in_figure(ax):
    if ax is None:
        return 0.10, 0.90, 0.10
    bb = ax.get_position()
    return float(bb.x0), float(bb.x1), float(bb.y0)


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


def _place_adaptive_bottom_legend(fig, ax, handles, labels, *,
                                  target_rows=3, fontsize=None, y_gap=0.040):
    if not handles or not labels:
        return None
    fs = _FS_LEGEND if fontsize is None else float(fontsize)
    x0, x1, _ = _axis_span_in_figure(ax)
    width = max(0.15, x1 - x0)
    ncol = _legend_ncol_by_width(fig, labels, width, target_rows, fs)
    n_rows = int(math.ceil(len(labels) / float(max(1, ncol))))
    bottom_margin = min(
        0.42,
        max(0.12, 0.05 + 0.052 * n_rows) + max(0.0, float(_LINE_LEGEND_BOTTOM_MARGIN_EXTRA)),
    )
    fig.subplots_adjust(bottom=bottom_margin)
    x0, x1, y0 = _axis_span_in_figure(ax)
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


def plot_sit_key_month_ano(sit_metric_all, model_labels, year_range, month,
                           hms=None, fig_name=None, line_style=None, color=None,
                           model_spread_payloads=None, **kwargs):
    """Plot key-month sea-ice volume anomaly series for obs and models."""
    mtag = _month_tag(month)
    mlabel = _month_tag(month).capitalize()

    if not sit_metric_all:
        return

    fig, ax = plt.subplots(1, 1, figsize=_line_figsize(10, 4))
    ax.axhline(0, **_line_zero_style(alpha=0.7))
    labels_seq = list(model_labels) if model_labels is not None else []
    group_rank: Dict[str, int] = {}
    for lb in labels_seq:
        if _is_group_label(lb) and lb not in group_rank:
            group_rank[lb] = len(group_rank)
    n_obs = 0
    for lbl in labels_seq[:len(sit_metric_all)]:
        if _is_obs_label(lbl):
            n_obs += 1
        else:
            break
    n_obs = max(1, n_obs)

    for jj, metric_dict in enumerate(sit_metric_all):
        x, y = _extract_key_month_series(
            metric_dict, f'Vol_ano_{mtag}', month=month, year_range=year_range,
        )
        if y.size == 0:
            continue
        label = model_labels[jj] if jj < len(model_labels) else f'dataset{jj + 1}'
        if jj < n_obs:
            style = _obs_style(jj)
            ax.plot(x, y, label=label, **style)
        elif _is_group_label(label):
            style = _group_style(group_rank.get(label, 0))
            ax.plot(x, y, label=label, **style)
        else:
            ls, cr = _get_style(jj, n_obs, line_style, color)
            style = {'linestyle': ls, 'color': cr, 'lw': _line_model_width(0.8)}
            ax.plot(x, y, linestyle=ls, color=cr, lw=_line_model_width(0.8), label=label, **kwargs)
        if jj >= n_obs and isinstance(model_spread_payloads, (list, tuple)):
            spread_idx = jj - n_obs
            if spread_idx < len(model_spread_payloads):
                spread_metric = model_spread_payloads[spread_idx] or {}
                xs, ys = _extract_key_month_series(
                    spread_metric, f'Vol_ano_{mtag}', month=month, year_range=year_range,
                )
                if ys.size > 0:
                    n_use = min(y.size, ys.size)
                    x_use = np.asarray(x[:n_use]) if np.asarray(x).size >= n_use else np.asarray(xs[:n_use])
                    _plot_group_std_band(
                        ax,
                        x_use,
                        np.asarray(y[:n_use], dtype=float),
                        np.asarray(ys[:n_use], dtype=float),
                        style,
                    )

    ax.set_title(f'Sea ice volume anomaly ({mlabel})', fontsize=_FS_SUBPLOT_TITLE)
    ax.set_ylabel(r'$10^3\ km^3$', fontsize=_FS_AXIS_LABEL)
    ax.set_xlabel('Year', fontsize=_FS_AXIS_LABEL)
    _apply_date_ticks(ax, minticks=4, maxticks=8)
    ax.tick_params(axis='x', rotation=45, labelsize=_FS_TICK)
    ax.tick_params(axis='y', labelsize=_FS_TICK)
    _apply_light_grid(ax)
    handles, labels_leg = ax.get_legend_handles_labels()
    _finalize_line_layout(fig, legend_location='none', pad=0.86)
    _place_adaptive_bottom_legend(
        fig, ax, handles, labels_leg,
        target_rows=3,
        fontsize=max(10, _FS_LEGEND),
        y_gap=0.056,
    )
    _save_fig(fig_name)


def plot_snd_key_month_ano(snd_metric_all, model_labels, year_range, month,
                           hms=None, fig_name=None, line_style=None, color=None,
                           model_spread_payloads=None, **kwargs):
    """Plot key-month snow-volume anomaly series for obs and models."""
    mtag = _month_tag(month)
    mlabel = _month_tag(month).capitalize()

    if not snd_metric_all:
        return

    fig, ax = plt.subplots(1, 1, figsize=_line_figsize(10, 4))
    ax.axhline(0, **_line_zero_style(alpha=0.7))
    labels_seq = list(model_labels) if model_labels is not None else []
    group_rank: Dict[str, int] = {}
    for lb in labels_seq:
        if _is_group_label(lb) and lb not in group_rank:
            group_rank[lb] = len(group_rank)
    n_obs = 0
    for lbl in labels_seq[:len(snd_metric_all)]:
        if _is_obs_label(lbl):
            n_obs += 1
        else:
            break
    n_obs = max(1, n_obs)

    for jj, metric_dict in enumerate(snd_metric_all):
        x, y = _extract_key_month_series(
            metric_dict, f'Vol_ano_{mtag}', month=month, year_range=year_range,
        )
        if y.size == 0:
            continue
        label = model_labels[jj] if jj < len(model_labels) else f'dataset{jj + 1}'
        if jj < n_obs:
            style = _obs_style(jj)
            ax.plot(x, y, label=label, **style)
        elif _is_group_label(label):
            style = _group_style(group_rank.get(label, 0))
            ax.plot(x, y, label=label, **style)
        else:
            ls, cr = _get_style(jj, n_obs, line_style, color)
            style = {'linestyle': ls, 'color': cr, 'lw': _line_model_width(0.8)}
            ax.plot(x, y, linestyle=ls, color=cr, lw=_line_model_width(0.8), label=label, **kwargs)
        if jj >= n_obs and isinstance(model_spread_payloads, (list, tuple)):
            spread_idx = jj - n_obs
            if spread_idx < len(model_spread_payloads):
                spread_metric = model_spread_payloads[spread_idx] or {}
                xs, ys = _extract_key_month_series(
                    spread_metric, f'Vol_ano_{mtag}', month=month, year_range=year_range,
                )
                if ys.size > 0:
                    n_use = min(y.size, ys.size)
                    x_use = np.asarray(x[:n_use]) if np.asarray(x).size >= n_use else np.asarray(xs[:n_use])
                    _plot_group_std_band(
                        ax,
                        x_use,
                        np.asarray(y[:n_use], dtype=float),
                        np.asarray(ys[:n_use], dtype=float),
                        style,
                    )

    ax.set_title(f'Snow volume anomaly ({mlabel})', fontsize=_FS_SUBPLOT_TITLE)
    ax.set_ylabel(r'$10^3\ km^3$', fontsize=_FS_AXIS_LABEL)
    ax.set_xlabel('Year', fontsize=_FS_AXIS_LABEL)
    _apply_date_ticks(ax, minticks=4, maxticks=8)
    ax.tick_params(axis='x', rotation=45, labelsize=_FS_TICK)
    ax.tick_params(axis='y', labelsize=_FS_TICK)
    _apply_light_grid(ax)
    handles, labels_leg = ax.get_legend_handles_labels()
    _finalize_line_layout(fig, legend_location='none', pad=0.86)
    _place_adaptive_bottom_legend(
        fig, ax, handles, labels_leg,
        target_rows=3,
        fontsize=max(10, _FS_LEGEND),
        y_gap=0.056,
    )
    _save_fig(fig_name)


def plot_sit_ts(SIT_AllModel, SIT_DIFF_AllModel,
                model_labels, fig_name,
                line_style=None, color=None, model_spread_payloads=None, **kwargs):
    """Plot the seasonal cycle of integrated sea-ice volume.

    This function intentionally uses *two* inputs:

      - ``SIT_DIFF_AllModel`` provides the observation payloads, because obs1 and
        obs2 are stored alongside pairwise comparison dictionaries in the main
        orchestration layer.
      - ``SIT_AllModel`` provides the model-only metrics, so full-grid model
        curves remain independent from any obs-masked pairwise bookkeeping.

    This separation keeps model curves independent from pairwise comparison bookkeeping.

    Args:
        SIT_AllModel:       List of model metric dictionaries.
        SIT_DIFF_AllModel:  List containing observation payloads and/or paired
                            comparison dictionaries used to recover obs curves.
        model_labels:       Labels in observation-first order.
        fig_name:           Output file path.
    """
    if not SIT_DIFF_AllModel:
        _save_fig(fig_name)
        return

    first = SIT_DIFF_AllModel[0]
    has_obs2 = ('yearmon_list' not in first) and (len(model_labels) >= 2)
    group_rank: Dict[str, int] = {}
    for lb in (model_labels or []):
        if _is_group_label(lb) and lb not in group_rank:
            group_rank[lb] = len(group_rank)

    obs1 = _extract_monthly_cycle(first.get('thick1_metric') or {}, 'Vol_clim')
    if obs1 is None or np.all(np.isnan(obs1)):
        logger.warning('plot_sit_ts: obs1 unavailable, skipping.')
        _save_fig(fig_name)
        return

    fig, ax = plt.subplots(1, 1, figsize=_line_figsize(11, 4.6))
    month_dates = _month_cycle_dates()
    _plot_monthly_cycle(ax, obs1, _obs_style(0), (model_labels[0] if model_labels else 'obs1'), x=month_dates)

    if has_obs2:
        obs2 = _extract_monthly_cycle(first.get('thick2_metric') or {}, 'Vol_clim')
        if obs2 is not None and not np.all(np.isnan(obs2)):
            _plot_monthly_cycle(ax, obs2, _obs_style(1), model_labels[1], x=month_dates)

    label_start = 2 if has_obs2 else 1
    for model_idx, metric_model in enumerate(SIT_AllModel):
        y = _extract_monthly_cycle(metric_model, 'Vol_clim')

        if y is None or np.all(np.isnan(y)):
            continue

        model_name = model_labels[label_start + model_idx] if (model_labels and label_start + model_idx < len(model_labels)) else f'model{model_idx + 1}'
        style_idx = label_start + model_idx
        if _is_group_label(model_name):
            style = _group_style(group_rank.get(model_name, 0))
            _plot_monthly_cycle(
                ax, y,
                style,
                model_name,
                x=month_dates,
            )
        else:
            ls, cr = _get_style(style_idx, label_start, line_style, color)
            style = {'color': cr, 'linestyle': ls, 'linewidth': _line_model_width(0.9)}
            _plot_monthly_cycle(
                ax, y,
                style,
                model_name,
                x=month_dates,
            )
        if isinstance(model_spread_payloads, (list, tuple)) and model_idx < len(model_spread_payloads):
            spread_metric = model_spread_payloads[model_idx] or {}
            ys = _extract_monthly_cycle(spread_metric, 'Vol_clim')
            if ys is not None and not np.all(np.isnan(ys)):
                _plot_group_std_band(
                    ax,
                    month_dates,
                    np.asarray(y, dtype=float),
                    np.asarray(ys, dtype=float),
                    style,
                )

    _apply_month_ticks(ax, month_dates=month_dates, interval=2, rotation=30, use_datetime=True)
    ax.tick_params(axis='y', labelsize=_FS_TICK)
    _apply_light_grid(ax)
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.set_ylim(bottom=0)
    ax.set_ylabel(r'Sea ice volume ($10^3$ $km^3$)', fontsize=_FS_AXIS_LABEL)

    _finalize_line_layout(fig, legend_location='none', pad=0.85)
    _place_adaptive_bottom_legend(
        fig, ax, handles, labels_leg,
        target_rows=3,
        fontsize=max(10, _FS_LEGEND),
        y_gap=0.054,
    )
    _save_fig(fig_name)

def plot_sit_ano(SIT_AllModel, SIT_DIFF_AllModel, model_labels, year_range,
                 hms=None, line_style=None, color=None, fig_name=None,
                 model_spread_payloads=None):
    """Plot sea-ice volume anomaly time series."""
    if not SIT_DIFF_AllModel:
        _save_fig(fig_name)
        return

    fig, ax = plt.subplots(1, 1, figsize=_line_figsize(12, 5))
    ref_dates = pd.date_range(f'{int(year_range[0]):04d}-01-01', f'{int(year_range[1]):04d}-12-01', freq='MS')
    ax.plot(ref_dates, np.zeros(ref_dates.size), **_line_zero_style(alpha=0.7))
    ax.set_xlim(datetime.datetime(int(year_range[0]), 1, 1), datetime.datetime(int(year_range[1]), 12, 31))

    first = SIT_DIFF_AllModel[0]
    has_obs2 = len(model_labels) >= (len(SIT_AllModel) + 2)
    group_rank: Dict[str, int] = {}
    for lb in (model_labels or []):
        if _is_group_label(lb) and lb not in group_rank:
            group_rank[lb] = len(group_rank)

    obs1_metric = first.get('thick1_metric') or {}
    obs1_dates, obs1_vals = _extract_series_with_dates(
        obs1_metric, 'Vol_ano', year_range=year_range, yearmon_list=first.get('yearmon_list', []),
    )
    _plot_anomaly_timeseries(
        ax, obs1_dates, obs1_vals, _obs_style(0),
        (model_labels[0] if len(model_labels) > 0 else 'obs1'),
    )

    if has_obs2:
        obs2_metric = first.get('thick2_metric') or {}
        obs2_dates, obs2_vals = _extract_series_with_dates(
            obs2_metric, 'Vol_ano', year_range=year_range, yearmon_list=first.get('yearmon_list', []),
        )
        _plot_anomaly_timeseries(
            ax, obs2_dates, obs2_vals, _obs_style(1),
            (model_labels[1] if len(model_labels) > 1 else 'obs2'),
        )

    model_entries = SIT_DIFF_AllModel[1:] if has_obs2 else SIT_DIFF_AllModel
    for model_idx, d in enumerate(model_entries):
        metric_model = d.get('thick2_metric') or (
            SIT_AllModel[model_idx] if model_idx < len(SIT_AllModel) else {}
        )
        dates_m, vals_m = _extract_series_with_dates(
            metric_model, 'Vol_ano', year_range=year_range, yearmon_list=d.get('yearmon_list', []),
        )
        if vals_m.size == 0:
            continue
        label_idx = model_idx + (2 if has_obs2 else 1)
        n_obs = 2 if has_obs2 else 1
        model_name = model_labels[label_idx] if label_idx < len(model_labels) else f'model{model_idx + 1}'
        if _is_group_label(model_name):
            style = _group_style(group_rank.get(model_name, 0))
        else:
            ls, cr = _get_style(label_idx, n_obs, line_style, color)
            style = {'color': cr, 'linestyle': ls, 'linewidth': _line_model_width(0.9)}
        _plot_anomaly_timeseries(ax, dates_m, vals_m, style, model_name)
        if isinstance(model_spread_payloads, (list, tuple)) and model_idx < len(model_spread_payloads):
            spread_metric = model_spread_payloads[model_idx] or {}
            dates_s, vals_s = _extract_series_with_dates(
                spread_metric, 'Vol_ano', year_range=year_range, yearmon_list=spread_metric.get('yearmon_list', []),
            )
            if vals_s.size > 0:
                n_use = min(vals_m.size, vals_s.size)
                x_use = np.asarray(dates_m[:n_use]) if np.asarray(dates_m).size >= n_use else np.asarray(dates_s[:n_use])
                _plot_group_std_band(
                    ax,
                    x_use,
                    np.asarray(vals_m[:n_use], dtype=float),
                    np.asarray(vals_s[:n_use], dtype=float),
                    style,
                )

    _apply_date_ticks(ax, minticks=4, maxticks=8)
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.tick_params(axis='x', rotation=45, labelsize=_FS_TICK)
    ax.tick_params(axis='y', labelsize=_FS_TICK)
    ax.set_ylabel(r'$10^3\ km^3$', fontsize=_FS_AXIS_LABEL)
    _apply_light_grid(ax)

    _finalize_line_layout(fig, legend_location='none', pad=0.86)
    _place_adaptive_bottom_legend(
        fig, ax, handles, labels_leg,
        target_rows=3,
        fontsize=max(10, _FS_LEGEND),
        y_gap=0.056,
    )
    _save_fig(fig_name)

def plot_sit_map(lon, lat, SIT_DIFF_AllModel, mon, hms, data_range, diff_range, unit='m',
                 cm='viridis', diff_cm='RdBu_r',
                 model_labels=None, plot_mode='mixed', fig_name=None, **kwargs):
    """Spatial climatology maps for sea ice thickness.

    Args:
        plot_mode: 'mixed' (default), 'raw' (all raw), or 'diff' (all diffs vs obs1).
    """
    if not SIT_DIFF_AllModel:
        return

    # --- reference (obs1) ---
    ref = (SIT_DIFF_AllModel[0].get('thick1_metric') or {})
    ref_mon = np.array(ref.get('uni_mon', []), dtype=int)
    ref_clim = ref.get('thick_clim')

    if ref_clim is None or ref_mon.size == 0:
        return
    if mon not in ref_mon:
        return

    ref_idx = int(np.where(ref_mon == mon)[0][0])
    ref_field = np.array(ref_clim)[ref_idx, :, :]
    if ref_field.ndim != 2:
        return

    rows_fields = [ref_field]
    labels_used = [model_labels[0] if model_labels else 'obs']

    # --- subsequent datasets (obs2 and models) ---
    for ii, d in enumerate(SIT_DIFF_AllModel):
        m = (d.get('thick2_metric') or {})
        uni = np.array(m.get('uni_mon', []), dtype=int)
        clim = m.get('thick_clim')
        if clim is None or uni.size == 0 or mon not in uni:
            continue
        idx = int(np.where(uni == mon)[0][0])
        field = np.array(clim)[idx, :, :]
        if field.ndim != 2 or field.shape != ref_field.shape:
            continue
        rows_fields.append(field)
        label_idx = ii + 1
        labels_used.append(model_labels[label_idx] if (model_labels and label_idx < len(model_labels)) else f'dataset{label_idx}')

    all_data = np.stack(rows_fields, axis=0)  # (N_total, nx, ny)
    N_total = all_data.shape[0]

    # Determine panels based on plot_mode
    if plot_mode == 'diff':
        if N_total < 2:
            return
        # First panel = obs1 raw, remaining = differences vs obs1
        diff_part = all_data[1:] - all_data[0:1]
        plot_data = np.concatenate([all_data[0:1], diff_part], axis=0)
        plot_labels = labels_used
    elif plot_mode == 'raw':
        plot_data = all_data
        plot_labels = labels_used
    else:
        plot_data = all_data
        plot_labels = labels_used

    N = plot_data.shape[0]
    if N == 0:
        return

    rows, cols = _adaptive_grid(N)
    proj = ccrs.Stereographic(central_latitude=-90 if hms == 'sh' else 90, central_longitude=0)

    fig, ax = plt.subplots(rows, cols, figsize=_map_figsize(cols, rows), subplot_kw={'projection': proj})
    ax = ax.flatten()

    cmp_raw = plt.get_cmap(cm, 10)
    cmp_diff = plt.get_cmap(diff_cm, 11)
    im_raw = None
    im_diff = None

    for ii in range(N):
        panel_label = plot_labels[ii] if ii < len(plot_labels) else f'dataset{ii + 1}'
        is_model_panel = not _is_obs_label(panel_label)
        field = np.array(plot_data[ii, :, :], dtype=float)

        if plot_mode == 'raw':
            if is_model_panel:
                field = _mask_model_zeros(field)
            im_raw = polar_map(hms, ax[ii]).pcolormesh(
                lon, lat, field,
                vmin=data_range[0], vmax=data_range[1],
                transform=ccrs.PlateCarree(), cmap=cmp_raw)
        elif plot_mode == 'diff':
            if ii == 0:
                # First panel: obs1 raw values
                im_raw = polar_map(hms, ax[ii]).pcolormesh(
                    lon, lat, field,
                    vmin=data_range[0], vmax=data_range[1],
                    transform=ccrs.PlateCarree(), cmap=cmp_raw)
            else:
                # Remaining panels: differences
                im_diff = polar_map(hms, ax[ii]).pcolormesh(
                    lon, lat, field,
                    vmin=diff_range[0], vmax=diff_range[1],
                    transform=ccrs.PlateCarree(), cmap=cmp_diff)
        else:  # mixed
            if ii == 0:
                im_raw = polar_map(hms, ax[ii]).pcolormesh(
                    lon, lat, field,
                    vmin=data_range[0], vmax=data_range[1],
                    transform=ccrs.PlateCarree(), cmap=cmp_raw)
            else:
                im_diff = polar_map(hms, ax[ii]).pcolormesh(
                    lon, lat, field - all_data[0, :, :],
                    vmin=diff_range[0], vmax=diff_range[1],
                    transform=ccrs.PlateCarree(), cmap=cmp_diff)
        ax[ii].set_title(panel_label, fontsize=_FS_SUBPLOT_TITLE)

    for j in range(N, len(ax)):
        fig.delaxes(ax[j])

    _finalize_map_layout(fig, has_bottom_cbar=False)
    # Colorbars
    # Colorbars at the bottom to avoid overlapping subplots
    if plot_mode == 'raw':
        _bottom_cbar(fig, im_raw, [ax[i] for i in range(N)], label=unit)
    elif plot_mode == 'diff':
        # Diff-focused panel: keep one shared difference colorbar.
        if im_diff is not None:
            _bottom_cbar(fig, im_diff, [ax[i] for i in range(N)],
                         label=unit, extend='auto')
        elif im_raw is not None:
            _bottom_cbar(fig, im_raw, [ax[0]], label=unit)
    else:  # mixed
        # Mixed panel: prioritize the difference colorbar for readability.
        if im_diff is not None:
            _bottom_cbar(fig, im_diff, [ax[i] for i in range(N)],
                         label=unit, extend='auto')
        elif im_raw is not None:
            _bottom_cbar(fig, im_raw, [ax[0]], label=unit)

    _save_fig(fig_name)

def plot_snd_ts(SND_AllModel, SND_DIFF_AllModel,
                model_labels, fig_name,
                line_style=None, color=None, model_spread_payloads=None, **kwargs):
    """Plot the seasonal cycle of integrated snow volume.

    The snow-depth workflow mirrors the thickness workflow: observation curves
    are recovered from the paired payload list, while model curves are read from
    the independent model-metric list supplied via ``SND_AllModel``.

    This design keeps model curves independent from pairwise comparison bookkeeping.

    Args:
        SND_AllModel:       List of model metric dictionaries.
        SND_DIFF_AllModel:  Observation-containing payload list used for obs1/obs2.
        model_labels:       Labels in observation-first order.
        fig_name:           Output file path.
    """
    if not SND_DIFF_AllModel:
        _save_fig(fig_name)
        return

    first = SND_DIFF_AllModel[0]
    has_obs2 = ('yearmon_list' not in first) and (len(model_labels) >= 2)
    group_rank: Dict[str, int] = {}
    for lb in (model_labels or []):
        if _is_group_label(lb) and lb not in group_rank:
            group_rank[lb] = len(group_rank)

    obs1 = _extract_monthly_cycle(first.get('thick1_metric') or {}, 'Vol_clim')
    if obs1 is None or np.all(np.isnan(obs1)):
        logger.warning('plot_snd_ts: obs1 unavailable, skipping.')
        _save_fig(fig_name)
        return

    fig, ax = plt.subplots(1, 1, figsize=_line_figsize(11, 4.6))
    month_dates = _month_cycle_dates()
    _plot_monthly_cycle(ax, obs1, _obs_style(0), (model_labels[0] if model_labels else 'obs1'), x=month_dates)

    if has_obs2:
        obs2 = _extract_monthly_cycle(first.get('thick2_metric') or {}, 'Vol_clim')
        if obs2 is not None and not np.all(np.isnan(obs2)):
            _plot_monthly_cycle(ax, obs2, _obs_style(1), model_labels[1], x=month_dates)
        else:
            logger.warning('plot_snd_ts: obs2 unavailable.')

    label_start = 2 if has_obs2 else 1
    for model_idx, metric_model in enumerate(SND_AllModel):
        y = _extract_monthly_cycle(metric_model, 'Vol_clim')

        if y is None or np.all(np.isnan(y)):
            continue

        model_name = model_labels[label_start + model_idx] if (model_labels and label_start + model_idx < len(model_labels)) else f'model{model_idx + 1}'
        style_idx = label_start + model_idx
        if _is_group_label(model_name):
            style = _group_style(group_rank.get(model_name, 0))
            _plot_monthly_cycle(
                ax, y,
                style,
                model_name,
                x=month_dates,
            )
        else:
            ls, cr = _get_style(style_idx, label_start, line_style, color)
            style = {'color': cr, 'linestyle': ls, 'linewidth': _line_model_width(0.9)}
            _plot_monthly_cycle(
                ax, y,
                style,
                model_name,
                x=month_dates,
            )
        if isinstance(model_spread_payloads, (list, tuple)) and model_idx < len(model_spread_payloads):
            spread_metric = model_spread_payloads[model_idx] or {}
            ys = _extract_monthly_cycle(spread_metric, 'Vol_clim')
            if ys is not None and not np.all(np.isnan(ys)):
                _plot_group_std_band(
                    ax,
                    month_dates,
                    np.asarray(y, dtype=float),
                    np.asarray(ys, dtype=float),
                    style,
                )

    _apply_month_ticks(ax, month_dates=month_dates, interval=2, rotation=30, use_datetime=True)
    ax.tick_params(axis='y', labelsize=_FS_TICK)
    _apply_light_grid(ax)
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.set_ylim(bottom=0)
    ax.set_ylabel(r'Snow volume ($10^3$ $km^3$)', fontsize=_FS_AXIS_LABEL)

    _finalize_line_layout(fig, legend_location='none', pad=0.85)
    _place_adaptive_bottom_legend(
        fig, ax, handles, labels_leg,
        target_rows=3,
        fontsize=max(10, _FS_LEGEND),
        y_gap=0.054,
    )
    _save_fig(fig_name)

def plot_snd_ano(SIT_AllModel, SIT_DIFF_AllModel, model_labels, year_range, line_style=None, color=None, fig_name=None,
                 model_spread_payloads=None):
    """Plot snow-volume anomaly time series."""
    if not SIT_DIFF_AllModel:
        _save_fig(fig_name)
        return

    fig, ax = plt.subplots(1, 1, figsize=_line_figsize(12, 5))
    ref_dates = pd.date_range(f'{int(year_range[0]):04d}-01-01', f'{int(year_range[1]):04d}-12-01', freq='MS')
    ax.plot(ref_dates, np.zeros(ref_dates.size), **_line_zero_style(alpha=0.7))
    ax.set_xlim(datetime.datetime(int(year_range[0]), 1, 1), datetime.datetime(int(year_range[1]), 12, 31))

    first = SIT_DIFF_AllModel[0]
    has_obs2 = len(model_labels) >= (len(SIT_AllModel) + 2)
    group_rank: Dict[str, int] = {}
    for lb in (model_labels or []):
        if _is_group_label(lb) and lb not in group_rank:
            group_rank[lb] = len(group_rank)

    obs1_metric = first.get('thick1_metric') or {}
    obs1_dates, obs1_vals = _extract_series_with_dates(
        obs1_metric, 'Vol_ano', year_range=year_range, yearmon_list=first.get('yearmon_list', []),
    )
    _plot_anomaly_timeseries(
        ax, obs1_dates, obs1_vals, _obs_style(0),
        (model_labels[0] if len(model_labels) > 0 else 'obs1'),
    )

    if has_obs2:
        obs2_metric = first.get('thick2_metric') or {}
        obs2_dates, obs2_vals = _extract_series_with_dates(
            obs2_metric, 'Vol_ano', year_range=year_range, yearmon_list=first.get('yearmon_list', []),
        )
        _plot_anomaly_timeseries(
            ax, obs2_dates, obs2_vals, _obs_style(1),
            (model_labels[1] if len(model_labels) > 1 else 'obs2'),
        )

    model_entries = SIT_DIFF_AllModel[1:] if has_obs2 else SIT_DIFF_AllModel
    for model_idx, d in enumerate(model_entries):
        metric_model = d.get('thick2_metric') or (
            SIT_AllModel[model_idx] if model_idx < len(SIT_AllModel) else {}
        )
        dates_m, vals_m = _extract_series_with_dates(
            metric_model, 'Vol_ano', year_range=year_range, yearmon_list=d.get('yearmon_list', []),
        )
        if vals_m.size == 0:
            continue
        label_idx = model_idx + (2 if has_obs2 else 1)
        n_obs = 2 if has_obs2 else 1
        model_name = model_labels[label_idx] if label_idx < len(model_labels) else f'model{model_idx + 1}'
        if _is_group_label(model_name):
            style = _group_style(group_rank.get(model_name, 0))
        else:
            ls, cr = _get_style(label_idx, n_obs, line_style, color)
            style = {'color': cr, 'linestyle': ls, 'linewidth': _line_model_width(0.9)}
        _plot_anomaly_timeseries(ax, dates_m, vals_m, style, model_name)
        if isinstance(model_spread_payloads, (list, tuple)) and model_idx < len(model_spread_payloads):
            spread_metric = model_spread_payloads[model_idx] or {}
            dates_s, vals_s = _extract_series_with_dates(
                spread_metric, 'Vol_ano', year_range=year_range, yearmon_list=spread_metric.get('yearmon_list', []),
            )
            if vals_s.size > 0:
                n_use = min(vals_m.size, vals_s.size)
                x_use = np.asarray(dates_m[:n_use]) if np.asarray(dates_m).size >= n_use else np.asarray(dates_s[:n_use])
                _plot_group_std_band(
                    ax,
                    x_use,
                    np.asarray(vals_m[:n_use], dtype=float),
                    np.asarray(vals_s[:n_use], dtype=float),
                    style,
                )

    _apply_date_ticks(ax, minticks=4, maxticks=8)
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.tick_params(axis='x', rotation=45, labelsize=_FS_TICK)
    ax.tick_params(axis='y', labelsize=_FS_TICK)
    ax.set_ylabel(r'$10^3\ km^3$', fontsize=_FS_AXIS_LABEL)
    _apply_light_grid(ax)

    _finalize_line_layout(fig, legend_location='none', pad=0.86)
    _place_adaptive_bottom_legend(
        fig, ax, handles, labels_leg,
        target_rows=3,
        fontsize=max(10, _FS_LEGEND),
        y_gap=0.056,
    )
    _save_fig(fig_name)

def plot_snd_map(lon, lat, SND_DIFF_AllModel, mon, hms, data_range, diff_range, unit='m',
                 cm='Blues', diff_cm='RdBu_r',
                 model_labels=None, plot_mode='mixed', fig_name=None, **kwargs):
    """Spatial climatology maps for snow depth (Feb or Sep).

    Args:
        plot_mode: 'mixed' (default), 'raw' (all raw), or 'diff' (all diffs vs obs1).
    """
    if not SND_DIFF_AllModel:
        return

    # --- reference (obs1) ---
    ref = (SND_DIFF_AllModel[0].get('thick1_metric') or {})
    ref_mon = np.array(ref.get('uni_mon', []), dtype=int)
    ref_clim = ref.get('thick_clim')

    if ref_clim is None or ref_mon.size == 0:
        return
    if mon not in ref_mon:
        return

    ref_idx = int(np.where(ref_mon == mon)[0][0])
    ref_field = np.array(ref_clim)[ref_idx, :, :]
    if ref_field.ndim != 2:
        return

    rows_fields = [ref_field]
    labels_used = [model_labels[0] if model_labels else 'obs']

    # --- subsequent datasets (obs2 and models) ---
    for ii, d in enumerate(SND_DIFF_AllModel):
        m = (d.get('thick2_metric') or {})
        uni = np.array(m.get('uni_mon', []), dtype=int)
        clim = m.get('thick_clim')
        if clim is None or uni.size == 0 or mon not in uni:
            continue
        idx = int(np.where(uni == mon)[0][0])
        field = np.array(clim)[idx, :, :]
        if field.ndim != 2 or field.shape != ref_field.shape:
            continue
        rows_fields.append(field)
        label_idx = ii + 1
        labels_used.append(model_labels[label_idx] if (model_labels and label_idx < len(model_labels)) else f'dataset{label_idx}')

    all_data = np.stack(rows_fields, axis=0)  # (N_total, nx, ny)
    N_total = all_data.shape[0]

    # Determine panels based on plot_mode
    if plot_mode == 'diff':
        if N_total < 2:
            return
        # First panel = obs1 raw, remaining = differences vs obs1
        diff_part = all_data[1:] - all_data[0:1]
        plot_data = np.concatenate([all_data[0:1], diff_part], axis=0)
        plot_labels = labels_used
    elif plot_mode == 'raw':
        plot_data = all_data
        plot_labels = labels_used
    else:
        plot_data = all_data
        plot_labels = labels_used

    N = plot_data.shape[0]
    if N == 0:
        return

    rows, cols = _adaptive_grid(N)
    proj = ccrs.Stereographic(central_latitude=-90 if hms == 'sh' else 90, central_longitude=0)

    fig, ax = plt.subplots(rows, cols, figsize=_map_figsize(cols, rows), subplot_kw={'projection': proj})
    ax = ax.flatten()

    cmp_raw = plt.get_cmap(cm, 10)
    cmp_diff = plt.get_cmap(diff_cm, 11)
    im_raw = None
    im_diff = None

    for ii in range(N):
        panel_label = plot_labels[ii] if ii < len(plot_labels) else f'dataset{ii + 1}'
        is_model_panel = not _is_obs_label(panel_label)
        field = np.array(plot_data[ii, :, :], dtype=float)

        if plot_mode == 'raw':
            if is_model_panel:
                field = _mask_model_zeros(field)
            im_raw = polar_map(hms, ax[ii]).pcolormesh(
                lon, lat, field,
                vmin=data_range[0], vmax=data_range[1],
                transform=ccrs.PlateCarree(), cmap=cmp_raw)
        elif plot_mode == 'diff':
            if ii == 0:
                # First panel: obs1 raw values
                im_raw = polar_map(hms, ax[ii]).pcolormesh(
                    lon, lat, field,
                    vmin=data_range[0], vmax=data_range[1],
                    transform=ccrs.PlateCarree(), cmap=cmp_raw)
            else:
                # Remaining panels: differences
                im_diff = polar_map(hms, ax[ii]).pcolormesh(
                    lon, lat, field,
                    vmin=diff_range[0], vmax=diff_range[1],
                    transform=ccrs.PlateCarree(), cmap=cmp_diff)
        else:  # mixed
            if ii == 0:
                im_raw = polar_map(hms, ax[ii]).pcolormesh(
                    lon, lat, field,
                    vmin=data_range[0], vmax=data_range[1],
                    transform=ccrs.PlateCarree(), cmap=cmp_raw)
            else:
                im_diff = polar_map(hms, ax[ii]).pcolormesh(
                    lon, lat, field - all_data[0, :, :],
                    vmin=diff_range[0], vmax=diff_range[1],
                    transform=ccrs.PlateCarree(), cmap=cmp_diff)
        ax[ii].set_title(panel_label, fontsize=_FS_SUBPLOT_TITLE)

    for j in range(N, len(ax)):
        fig.delaxes(ax[j])

    _finalize_map_layout(fig, has_bottom_cbar=False)
    if plot_mode == 'raw':
        _bottom_cbar(fig, im_raw, [ax[i] for i in range(N)], label=unit)
    elif plot_mode == 'diff':
        # Diff-focused panel: keep one shared difference colorbar.
        if im_diff is not None:
            _bottom_cbar(fig, im_diff, [ax[i] for i in range(N)],
                         label=unit, extend='auto')
        elif im_raw is not None:
            _bottom_cbar(fig, im_raw, [ax[0]], label=unit)
    else:  # mixed
        # Mixed panel: prioritize the difference colorbar for readability.
        if im_diff is not None:
            _bottom_cbar(fig, im_diff, [ax[i] for i in range(N)],
                         label=unit, extend='auto')
        elif im_raw is not None:
            _bottom_cbar(fig, im_raw, [ax[0]], label=unit)

    _save_fig(fig_name)

__all__ = [
    "plot_sit_key_month_ano",
    "plot_snd_key_month_ano",
    "plot_sit_ts",
    "plot_sit_ano",
    "plot_sit_map",
    "plot_snd_ts",
    "plot_snd_ano",
    "plot_snd_map",
]
