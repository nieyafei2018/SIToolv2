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
        0.46,
        max(0.14, 0.07 + 0.056 * n_rows) + max(0.0, float(_LINE_LEGEND_BOTTOM_MARGIN_EXTRA)),
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


def plot_SID_ts(sid_metric_obs, sid_metric_AllModel, model_labels,
                line_style=None, color=None, fig_name=None,
                model_spread_payloads=None, **kwargs):
    """Plot the monthly climatological cycle of domain-mean MKE."""
    fig, ax = plt.subplots(1, 1, figsize=_line_figsize(11.5, 4.6))
    month_dates = _month_cycle_dates()
    group_rank: Dict[str, int] = {}
    for lb in (model_labels or []):
        if _is_group_label(lb) and lb not in group_rank:
            group_rank[lb] = len(group_rank)

    metric_key = 'MKE_ts_clim'

    # observations: obs1 black, obs2 gray
    for obs_idx, metric in enumerate(sid_metric_obs):
        y = np.asarray(metric.get(metric_key, np.array([])), dtype=float).squeeze()
        if y.size != 12:
            continue
        label = model_labels[obs_idx] if obs_idx < len(model_labels) else f'obs{obs_idx + 1}'
        style = _obs_style(obs_idx)
        _plot_monthly_cycle(ax, y, style, label, x=month_dates)

    # models: canonical color cycle, solid lines
    label_start = len(sid_metric_obs)
    for model_idx, metric in enumerate(sid_metric_AllModel):
        y = np.asarray(metric.get(metric_key, np.array([])), dtype=float).squeeze()
        if y.size != 12:
            continue
        label_idx = label_start + model_idx
        label = model_labels[label_idx] if label_idx < len(model_labels) else f'model{model_idx + 1}'
        if _is_group_label(label):
            style = _group_style(group_rank.get(label, 0))
        else:
            ls, cr = _get_style(label_idx, label_start, line_style, color)
            style = {
                'color': cr,
                'linestyle': ls,
                'linewidth': _line_model_width(0.9),
            }
        _plot_monthly_cycle(ax, y, style, label, x=month_dates)
        if isinstance(model_spread_payloads, (list, tuple)) and model_idx < len(model_spread_payloads):
            spread_metric = model_spread_payloads[model_idx] or {}
            ys = np.asarray(spread_metric.get(metric_key, np.array([])), dtype=float).squeeze()
            if ys.size == 12:
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
    ax.set_ylabel(r'$m^2/s^2$', fontsize=_FS_AXIS_LABEL)
    ax.set_title('Mean Kinetic Energy', fontsize=_FS_SUBPLOT_TITLE)

    _finalize_line_layout(fig, legend_location='none', pad=0.85)
    _place_adaptive_bottom_legend(
        fig, ax, handles, labels_leg,
        target_rows=3,
        fontsize=max(10, _FS_LEGEND),
        y_gap=0.038,
    )
    _save_fig(fig_name)

def plot_SID_ano(sid_metric_obs, sid_metric_AllModel, model_labels, year_range, fig_name,
                 hms=None, line_style=None, color=None,
                 model_spread_payloads=None, **kwargs):
    """Plot MKE anomaly time series with standardized observation/model styles."""
    all_metrics = list(sid_metric_obs) + list(sid_metric_AllModel)
    if not all_metrics:
        _save_fig(fig_name)
        return

    year_sta, year_end = year_range[0], year_range[1]
    dates_ref = pd.date_range(f'{year_sta}-01-01', periods=12 * (year_end - year_sta + 1), freq='MS')
    metric_key = 'MKE_ts_ano'

    fig, ax = plt.subplots(1, 1, figsize=_line_figsize(14, 5.4))
    ax.plot(dates_ref, np.zeros(dates_ref.shape), **_line_zero_style(color='gray', alpha=0.3))
    group_rank: Dict[str, int] = {}
    for lb in (model_labels or []):
        if _is_group_label(lb) and lb not in group_rank:
            group_rank[lb] = len(group_rank)

    n_obs = len(sid_metric_obs)
    for jj, metric in enumerate(all_metrics):
        dates, y = _extract_series_with_dates(
            metric,
            metric_key,
            year_range=year_range,
            yearmon_list=metric.get('yearmon_list'),
        )
        if y.size == 0:
            continue
        # Keep matched and raw anomaly series on the same full monthly axis.
        # Missing matched months are represented as NaN, so the visual span
        # remains identical to the raw panel while preserving data gaps.
        if year_range is not None:
            full_dates = pd.date_range(
                f'{int(year_sta):04d}-01-01',
                f'{int(year_end):04d}-12-01',
                freq='MS',
            )
            if (
                np.issubdtype(np.asarray(dates).dtype, np.datetime64)
                and y.size != full_dates.size
            ):
                y_full = np.full(full_dates.size, np.nan, dtype=float)
                month_to_idx = {
                    pd.Timestamp(ts).to_period('M'): idx
                    for idx, ts in enumerate(full_dates)
                }
                for ts, val in zip(pd.to_datetime(dates), y):
                    pos = month_to_idx.get(ts.to_period('M'))
                    if pos is not None:
                        y_full[pos] = val
                dates = np.asarray(full_dates, dtype='datetime64[ns]')
                y = y_full

        base_label = model_labels[jj] if jj < len(model_labels) else f'dataset{jj + 1}'
        label_str = base_label

        if jj < n_obs:
            style = _obs_style(jj)
        elif _is_group_label(label_str):
            style = _group_style(group_rank.get(label_str, 0))
        else:
            ls, cr = _get_style(jj, n_obs, line_style, color)
            style = {
                'color': cr,
                'linestyle': ls,
                'linewidth': _line_model_width(0.8),
                'alpha': _line_model_alpha(0.6),
            }
        _plot_anomaly_timeseries(ax, dates, y, style, label_str)
        if jj >= n_obs and isinstance(model_spread_payloads, (list, tuple)):
            spread_idx = jj - n_obs
            if spread_idx < len(model_spread_payloads):
                spread_metric = model_spread_payloads[spread_idx] or {}
                dates_s, ys = _extract_series_with_dates(
                    spread_metric,
                    metric_key,
                    year_range=year_range,
                    yearmon_list=spread_metric.get('yearmon_list'),
                )
                if ys.size > 0:
                    n_use = min(y.size, ys.size)
                    x_use = np.asarray(dates[:n_use]) if np.asarray(dates).size >= n_use else np.asarray(dates_s[:n_use])
                    _plot_group_std_band(
                        ax,
                        x_use,
                        np.asarray(y[:n_use], dtype=float),
                        np.asarray(ys[:n_use], dtype=float),
                        style,
                    )

    year_sta_i = int(year_sta)
    year_end_i = int(year_end)
    span_years = year_end_i - year_sta_i + 1
    if span_years <= 12:
        year_step = 1
    elif span_years <= 24:
        year_step = 2
    else:
        year_step = 5
    ax.set_xlim(datetime.datetime(year_sta_i, 1, 1), datetime.datetime(year_end_i, 12, 31))
    ax.xaxis.set_major_locator(mdates.YearLocator(base=year_step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45, labelsize=_FS_TICK)
    ax.tick_params(axis='y', labelsize=_FS_TICK)
    ax.set_ylabel(r'$m^2/s^2$', fontsize=_FS_AXIS_LABEL)
    ax.set_title('MKE anomaly', fontsize=_FS_SUBPLOT_TITLE)
    _apply_light_grid(ax)
    _set_symmetric_ylim(ax, pad_ratio=0.1)

    handles, labels_leg = ax.get_legend_handles_labels()
    if handles:
        n_items = len(handles)
        ncol = min(4, max(2, int(math.ceil(n_items / 5))))
        ax.legend(
            handles, labels_leg,
            loc='upper left',
            ncol=ncol,
            fontsize=max(8, _FS_LEGEND - 2),
            frameon=True,
            borderaxespad=0.6,
            handlelength=2.0,
            columnspacing=1.0,
        )

    _finalize_line_layout(fig, legend_location='none', pad=0.9)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.14, hspace=0.08)
    _save_fig(fig_name)


def plot_SID_key_month_ano(sid_metric_obs, sid_metric_AllModel, model_labels, year_range,
                           month, fig_name=None, hms=None, line_style=None, color=None,
                           legend_y_gap=None, model_spread_payloads=None, **kwargs):
    """Plot key-month MKE anomaly time series (March/September)."""
    all_metrics = list(sid_metric_obs) + list(sid_metric_AllModel)
    if not all_metrics:
        _save_fig(fig_name)
        return

    month = int(month)
    mlabel = _month_tag(month).capitalize()
    fig, ax = plt.subplots(1, 1, figsize=_line_figsize(12, 4.6))
    ax.axhline(0, **_line_zero_style(alpha=0.7))
    group_rank: Dict[str, int] = {}
    for lb in (model_labels or []):
        if _is_group_label(lb) and lb not in group_rank:
            group_rank[lb] = len(group_rank)

    n_obs = len(sid_metric_obs)
    for jj, metric in enumerate(all_metrics):
        x, y = _extract_key_month_series(
            metric, 'MKE_ts_ano', month=month, year_range=year_range,
        )
        if y.size == 0:
            continue
        label = model_labels[jj] if jj < len(model_labels) else f'dataset{jj + 1}'
        if jj < n_obs:
            style = _obs_style(jj)
        elif _is_group_label(label):
            style = _group_style(group_rank.get(label, 0))
        else:
            ls, cr = _get_style(jj, n_obs, line_style, color)
            style = {
                'color': cr,
                'linestyle': ls,
                'linewidth': _line_model_width(0.9),
            }
        _plot_anomaly_timeseries(ax, x, y, style, label)
        if jj >= n_obs and isinstance(model_spread_payloads, (list, tuple)):
            spread_idx = jj - n_obs
            if spread_idx < len(model_spread_payloads):
                spread_metric = model_spread_payloads[spread_idx] or {}
                xs, ys = _extract_key_month_series(
                    spread_metric,
                    'MKE_ts_ano',
                    month=month, year_range=year_range,
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

    ax.set_title(f'MKE anomaly ({mlabel})', fontsize=_FS_SUBPLOT_TITLE)
    ax.set_ylabel(r'$m^2/s^2$', fontsize=_FS_AXIS_LABEL)
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
        y_gap=float(0.056 if legend_y_gap is None else legend_y_gap),
    )
    _save_fig(fig_name)


def plot_MKE_map(lon, lat, data, model_labels, hms, plot_mode='mixed',
                 raw_cmap='viridis', diff_cmap='RdBu_r',
                 fig_name=None, **kwargs):
    """Adaptive row/column layout for MKE spatial maps.

    Args:
        data:         (N, nx, ny) array — obs in data[0], others in data[1:].
        model_labels: Length-N list of labels.
        hms:          Hemisphere ('sh' or 'nh').
        plot_mode:    'mixed' (default), 'raw' (all raw), or 'diff' (all diffs).
        fig_name:     Output file path.
    """
    N_total = data.shape[0]

    if plot_mode == 'diff':
        if N_total < 2:
            return
        # First panel = obs1 raw, remaining = differences vs obs1
        diff_part = data[1:] - data[0:1]
        plot_data = np.concatenate([data[0:1], diff_part], axis=0)
        plot_labels = list(model_labels)
    elif plot_mode == 'raw':
        plot_data = data
        plot_labels = list(model_labels)
    else:
        plot_data = data
        plot_labels = list(model_labels)

    N = plot_data.shape[0]
    if N == 0:
        return
    rows, cols = _adaptive_grid(N)

    proj = ccrs.Stereographic(central_latitude=-90 if hms == 'sh' else 90, central_longitude=0)

    fig, ax = plt.subplots(rows, cols, figsize=_map_figsize(cols, rows), subplot_kw={'projection': proj})
    ax = ax.flatten()

    im_raw = None
    im_diff = None

    for ii in range(N):
        mke = np.copy(plot_data[ii, :, :])
        panel_label = plot_labels[ii] if ii < len(plot_labels) else f'dataset{ii + 1}'
        is_model_panel = not _is_obs_label(panel_label)

        if plot_mode == 'raw':
            if is_model_panel:
                mke = _mask_model_zeros(mke)
            im_raw = polar_map(hms, ax[ii]).pcolormesh(
                lon, lat, mke, vmin=0, vmax=0.01,
                transform=ccrs.PlateCarree(), cmap=plt.get_cmap(raw_cmap))
        elif plot_mode == 'diff':
            if ii == 0:
                # First panel: obs1 raw values
                im_raw = polar_map(hms, ax[ii]).pcolormesh(
                    lon, lat, mke, vmin=0, vmax=0.01,
                    transform=ccrs.PlateCarree(), cmap=plt.get_cmap(raw_cmap))
            else:
                # Remaining panels: differences
                im_diff = polar_map(hms, ax[ii]).pcolormesh(
                    lon, lat, mke, vmin=-0.01, vmax=0.01,
                    transform=ccrs.PlateCarree(), cmap=plt.get_cmap(diff_cmap))
        else:  # mixed
            if ii == 0:
                im_raw = polar_map(hms, ax[ii]).pcolormesh(
                    lon, lat, mke, vmin=0, vmax=0.01,
                    transform=ccrs.PlateCarree(), cmap=plt.get_cmap(raw_cmap))
            else:
                im_diff = polar_map(hms, ax[ii]).pcolormesh(
                    lon, lat, mke - data[0, :, :], vmin=-0.01, vmax=0.01,
                    transform=ccrs.PlateCarree(), cmap=plt.get_cmap(diff_cmap))
        ax[ii].set_title(panel_label, fontsize=_FS_SUBPLOT_TITLE)

    for j in range(N, len(ax)):
        fig.delaxes(ax[j])

    _finalize_map_layout(fig, has_bottom_cbar=False)
    # Colorbars at the bottom to avoid overlapping subplots
    if plot_mode == 'raw':
        _bottom_cbar(fig, im_raw, [ax[i] for i in range(N)], label=r'$m^2/s^2$')
    elif plot_mode == 'diff':
        # Diff-focused panel: keep one shared difference colorbar.
        if im_diff is not None:
            _bottom_cbar(fig, im_diff, [ax[i] for i in range(N)],
                         label=r'$m^2/s^2$', extend='auto')
        elif im_raw is not None:
            _bottom_cbar(fig, im_raw, [ax[0]], label=r'$m^2/s^2$')
    else:  # mixed
        # Mixed panel: prioritize the difference colorbar for readability.
        if im_diff is not None:
            _bottom_cbar(fig, im_diff, [ax[i] for i in range(N)],
                         label=r'$m^2/s^2$', extend='auto')
        elif im_raw is not None:
            _bottom_cbar(fig, im_raw, [ax[0]], label=r'$m^2/s^2$')

    _save_fig(fig_name)

def plot_VectCorr_map(lon, lat, data, model_labels, hms, data_cm='viridis', unit='', fig_name=None, **kwargs):
    """

    Parameters
    ----------
    - data: 3-D array in shape of (N, nx, ny)
    - model_labels: A list or array of length N with the labels for the model data

    """
    N, nx, ny = data.shape
    rows, cols = _adaptive_grid(N)
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite)]
    auto_vmax = float(np.nanpercentile(finite, 95)) if finite.size else np.nan
    if not np.isfinite(auto_vmax):
        auto_vmax = 0.8
    auto_vmax = max(0.20, min(2.0, auto_vmax))
    vmin = float(kwargs.get('vmin', 0.0))
    vmax = float(kwargs.get('vmax', auto_vmax))

    proj = ccrs.Stereographic(central_latitude=-90 if hms == 'sh' else 90, central_longitude=0)

    fig, ax = plt.subplots(rows, cols, figsize=_map_figsize(cols, rows), subplot_kw={'projection': proj})
    ax = ax.flatten()
    for ii in range(N):
        panel_label = model_labels[ii] if ii < len(model_labels) else f'dataset{ii + 1}'
        corr = np.copy(data[ii, :, :])
        cmp = plt.get_cmap(data_cm, 10)
        im1 = polar_map(hms, ax[ii]).pcolormesh(
            lon, lat, corr,
            vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(), cmap=cmp,
        )
        ax[ii].set_title(panel_label, fontsize=_FS_SUBPLOT_TITLE)

    # Hide redundant axes
    for j in range(N, len(ax)):
        fig.delaxes(ax[j])

    _finalize_map_layout(fig, has_bottom_cbar=False)
    # Colorbar at the bottom to avoid overlapping subplots
    _bottom_cbar(fig, im1, [ax[i] for i in range(N)], label=unit, extend='auto')

    _save_fig(fig_name)

def plot_SID_map(lon, lat, data, model_labels, hms, kk,
                 sid_range=None, data_cm='YlGn', unit='',
                 rotate_flags=None, fig_name=None,
                 quiver_scale=1.0, quiver_width=0.004,
                 quiver_min_speed=0.0,
                 **kwargs):
    """Plot winter sea-ice drift speed with quiver vectors only."""
    N, _, _, nx, ny = data.shape
    rows, cols = _adaptive_grid(N)
    if sid_range is None:
        sid_range = [0, 0.3]
    kk = max(1, int(kk))
    quiver_width = max(0.0001, float(quiver_width))
    quiver_min_speed = max(0.0, float(quiver_min_speed))

    if rotate_flags is None:
        rotate_flags = [True] * N
    if len(rotate_flags) < N:
        rotate_flags = list(rotate_flags) + [True] * (N - len(rotate_flags))
    else:
        rotate_flags = list(rotate_flags[:N])

    proj = ccrs.Stereographic(
        central_latitude=-90 if hms == 'sh' else 90, central_longitude=0)
    points_proj = proj.transform_points(ccrs.PlateCarree(), lon, lat)
    x0, y0 = points_proj[..., 0], points_proj[..., 1]

    # Winter months: SH = JJA (indices 5,6,7); NH = DJF (indices 0,1,11)
    mon_index = [5, 6, 7] if hms == 'sh' else [0, 1, 11]

    fig, ax = plt.subplots(rows, cols, figsize=_map_figsize(cols, rows),
                           subplot_kw={'projection': proj})
    ax = ax.flatten()
    cmp = plt.get_cmap(data_cm, 10)

    for ii in range(N):
        panel_label = model_labels[ii] if ii < len(model_labels) else f'dataset{ii + 1}'
        is_model_panel = not _is_obs_label(panel_label)
        u_win = np.nanmean(data[ii, 0, mon_index, :, :], axis=0)
        v_win = np.nanmean(data[ii, 1, mon_index, :, :], axis=0)
        speed = np.sqrt(u_win ** 2 + v_win ** 2)
        if is_model_panel:
            speed = _mask_model_zeros(speed)

        u_quiver = np.array(u_win, dtype=float, copy=True)
        v_quiver = np.array(v_win, dtype=float, copy=True)
        if is_model_panel:
            vec_speed = np.sqrt(u_quiver ** 2 + v_quiver ** 2)
            valid = np.isfinite(speed) & np.isfinite(vec_speed)
            if quiver_min_speed > 0.0:
                valid &= (vec_speed > quiver_min_speed)
            u_quiver[~valid] = np.nan
            v_quiver[~valid] = np.nan

        polar_map(hms, ax[ii])
        im1 = ax[ii].pcolormesh(
            x0, y0, speed,
            vmin=sid_range[0], vmax=sid_range[1],
            cmap=cmp, transform=proj,
        )

        if rotate_flags[ii]:
            qx, qy = x0[::kk, ::kk], y0[::kk, ::kk]
            qtransform = proj
        else:
            qx, qy = lon[::kk, ::kk], lat[::kk, ::kk]
            qtransform = ccrs.PlateCarree()

        ax[ii].quiver(
            qx, qy, u_quiver[::kk, ::kk], v_quiver[::kk, ::kk],
            scale=quiver_scale, width=quiver_width, transform=qtransform,
        )
        ax[ii].set_title(panel_label, fontsize=_FS_SUBPLOT_TITLE)

    for j in range(N, len(ax)):
        fig.delaxes(ax[j])

    _finalize_map_layout(fig, has_bottom_cbar=False)
    _bottom_cbar(fig, im1, [ax[i] for i in range(N)],
                 label=unit if unit else r'm s$^{-1}$', extend='auto')
    _save_fig(fig_name)

__all__ = [
    "plot_SID_ts",
    "plot_SID_ano",
    "plot_SID_key_month_ano",
    "plot_MKE_map",
    "plot_VectCorr_map",
    "plot_SID_map",
]
