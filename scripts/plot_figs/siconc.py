# -*- coding: utf-8 -*-
"""Plotting routines split by diagnostic family."""

from scripts.plot_figs import core as _core
from scripts import utils

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
                                  target_rows=3, fontsize=None, y_gap=0.010):
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


def plot_siconc_key_month_ano(sic_metric_obs, sic_metric_models, model_labels,
                              year_range, month, hms=None, fig_name=None,
                              line_style=None, color=None, legend_y_gap=None,
                              model_spread_payloads=None, **kwargs):
    """Plot key-month SIC anomaly series (SIE/SIA/PIA/MIZ) for obs and models."""
    mtag = _month_tag(month)
    mlabel = _month_tag(month).capitalize()

    metrics_all = list(sic_metric_obs) + list(sic_metric_models)
    n_obs = len(sic_metric_obs)
    group_rank: dict = {}
    for lb in model_labels:
        if _is_group_label(lb) and lb not in group_rank:
            group_rank[lb] = len(group_rank)

    fig, ax = plt.subplots(4, 1, figsize=_line_figsize(12.4, 10.0), sharex=True)
    ax = ax.flatten()

    panels = [
        ('SIE', r'$10^6\ km^2$'),
        ('SIA', r'$10^6\ km^2$'),
        ('PIA', r'$10^6\ km^2$'),
        ('MIZ', r'$10^6\ km^2$'),
    ]
    title = [
        f'(a) SIE anomaly ({mlabel})',
        f'(b) SIA anomaly ({mlabel})',
        f'(c) PIA anomaly ({mlabel})',
        f'(d) MIZ anomaly ({mlabel})',
    ]

    for ii, (var, unit) in enumerate(panels):
        ax[ii].axhline(0, **_line_zero_style(alpha=0.7))
        for jj, metric_dict in enumerate(metrics_all):
            x, y = _extract_key_month_series(
                metric_dict, f'{var}_ano_{mtag}', month=month, year_range=year_range,
            )
            if y.size == 0:
                x, y = _extract_key_month_series(
                    metric_dict, f'{var}_ano', month=month, year_range=year_range,
                )
            if y.size == 0:
                continue
            label = model_labels[jj] if jj < len(model_labels) else f'dataset{jj + 1}'
            if jj < n_obs:
                style = _obs_style(jj)
                ax[ii].plot(x, y, label=label, **style)
            elif _is_group_label(label):
                style = _group_style(group_rank.get(label, 0))
                ax[ii].plot(
                    x, y, label=label,
                    **style,
                )
            else:
                ls, cr = _get_style(jj, n_obs, line_style, color)
                style = {
                    'linestyle': ls,
                    'color': cr,
                    'lw': _line_model_width(0.8),
                }
                ax[ii].plot(
                    x, y, linestyle=ls, color=cr,
                    lw=_line_model_width(0.8), label=label, **kwargs,
                )
            if jj >= n_obs and isinstance(model_spread_payloads, (list, tuple)):
                spread_idx = jj - n_obs
                if spread_idx < len(model_spread_payloads):
                    spread_dict = model_spread_payloads[spread_idx]
                    xs, ys = _extract_key_month_series(
                        spread_dict or {},
                        f'{var}_ano_{mtag}',
                        month=month, year_range=year_range,
                    )
                    if ys.size == 0:
                        xs, ys = _extract_key_month_series(
                            spread_dict or {},
                            f'{var}_ano',
                            month=month, year_range=year_range,
                        )
                    if ys.size > 0:
                        n_use = min(y.size, ys.size)
                        _plot_group_std_band(
                            ax[ii],
                            np.asarray(x[:n_use]),
                            np.asarray(y[:n_use], dtype=float),
                            np.asarray(ys[:n_use], dtype=float),
                            style,
                        )

        ax[ii].set_ylabel(unit, fontsize=_FS_AXIS_LABEL)
        ax[ii].tick_params(axis='y', labelsize=_FS_TICK)
        _apply_light_grid(ax[ii])
        ax[ii].text(
            -0.005, 1.03, title[ii],
            ha='left', va='bottom',
            transform=ax[ii].transAxes,
            fontsize=_FS_SUBPLOT_TITLE,
            clip_on=False,
        )

    ax[-1].set_xlabel('Year', fontsize=_FS_AXIS_LABEL)
    _apply_date_ticks(ax[-1], minticks=4, maxticks=8)
    ax[-1].tick_params(axis='x', rotation=45, labelsize=_FS_TICK)

    handles, labels_leg = ax[0].get_legend_handles_labels()
    if handles and labels_leg:
        fig.legend(
            handles, labels_leg,
            loc='center left', bbox_to_anchor=(0.80, 0.5),
            ncol=1, fontsize=max(11, _FS_LEGEND + 2),
            frameon=True, borderaxespad=0.0,
        )
    _finalize_line_layout(fig, legend_location='right', pad=0.78)
    fig.subplots_adjust(left=0.09, right=0.78, hspace=_line_subplot_hspace(0.34))
    _save_fig(fig_name)

def plot_heat_map(data, row_labels, col_labels, ax=None,
                  cbar_kw=None, cbarlabel="",
                  obs_row=None, obs_row_label='obs2',
                  ratio_vmin=0.5, ratio_vmax=2.0, cmap='RdBu_r',
                  add_colorbar=True, **kwargs):
    """Draw a reference-relative heatmap.

    When *obs_row* is provided the layout is:
      - Row 0: obs2 absolute values on a solid grey background.
      - Rows 1+: model/obs2 ratios coloured by a diverging colormap centered
        at 1 so values > 1 indicate larger error than the observational
        uncertainty.

    Args:
        data:           2-D array (N_models, M) of model metric values.
        row_labels:     Length-N_models list of model labels.
        col_labels:     Length-M list of metric labels.
        ax:             Matplotlib axes (defaults to current axes).
        cbar_kw:        Extra kwargs forwarded to ``fig.colorbar``.
        cbarlabel:      Colorbar label text.
        obs_row:        1-D array (M,) of absolute obs2 values (baseline row).
                        When provided the first row is rendered as the obs2 reference row.
        obs_row_label:  Label for the obs2 reference row.
        ratio_vmin/max: Color range for model ratio rows (default 0.5 – 2.0).
        add_colorbar:   Draw colorbar when True.

    Returns:
        The ``AxesImage`` produced by ``imshow``.
    """
    from matplotlib.patches import Rectangle

    if cbar_kw is None:
        cbar_kw = {}
    else:
        cbar_kw = dict(cbar_kw)
    if ax is None:
        ax = plt.gca()

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError('data must be a 2-D array')

    has_obs = obs_row is not None
    n_models, n_metrics = data.shape
    ratio_data = None

    if has_obs:
        obs_row = np.asarray(obs_row, dtype=float).reshape(-1)
        if obs_row.size != n_metrics:
            raise ValueError('obs_row size must match the number of heatmap columns')
        # Compute ratios: model absolute / obs2 absolute (guard against zero)
        obs2_safe = np.where(np.abs(obs_row) < 1e-9, 1e-9, np.abs(obs_row))
        ratio_data = np.abs(data) / obs2_safe[np.newaxis, :]
        # Full display array: NaN placeholder row for obs2 + ratio rows
        display_data = np.vstack([np.full((1, n_metrics), np.nan), ratio_data])
        all_labels = [obs_row_label] + list(row_labels)
        n_rows = n_models + 1
    else:
        display_data = data
        all_labels = list(row_labels)
        n_rows = n_models

    # --- Draw heatmap with diverging colormap and a clear center ---
    if has_obs:
        norm = matplotlib.colors.TwoSlopeNorm(
            vmin=ratio_vmin,
            vcenter=1.0,
            vmax=ratio_vmax,
        )
        im = ax.imshow(display_data, cmap=cmap, norm=norm, aspect='auto')
    else:
        finite_vals = display_data[np.isfinite(display_data)]
        if finite_vals.size == 0:
            vmin, vcenter, vmax = -1.0, 0.0, 1.0
        else:
            vmin = float(np.nanmin(finite_vals))
            vmax = float(np.nanmax(finite_vals))
            if np.isclose(vmin, vmax):
                delta = max(1.0, abs(vmin) * 0.1)
                vmin -= delta
                vmax += delta
            if vmin < 0 < vmax:
                vcenter = 0.0
            else:
                vcenter = (vmin + vmax) / 2.0
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        im = ax.imshow(display_data, cmap=cmap, norm=norm, aspect='auto')

    # --- Overlay obs2 row with solid grey background ---
    if has_obs:
        for j in range(n_metrics):
            ax.add_patch(Rectangle(
                (j - 0.5, -0.5), 1, 1,
                facecolor='#AAAAAA', edgecolor='white', linewidth=2, zorder=2,
            ))

    # --- Colorbar for model ratio rows ---
    cbar = None
    if add_colorbar:
        cbar_kw.setdefault('fraction', 0.026)
        cbar_kw.setdefault('pad', 0.014)
        cbar_kw.setdefault('aspect', 38)
        cbar_extend_req = cbar_kw.pop('extend', 'auto')
        cbar_extend = _resolve_colorbar_extend(
            im,
            data=(ratio_data if has_obs else display_data),
            vmin=getattr(norm, 'vmin', None),
            vmax=getattr(norm, 'vmax', None),
            extend=cbar_extend_req,
        )
        cbar = ax.figure.colorbar(im, ax=ax, extend=cbar_extend, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va='bottom', fontsize=max(16, _FS_CBAR_LABEL))
        cbar.ax.tick_params(labelsize=max(15, _FS_CBAR_TICK))

    # --- Tick labels ---
    ax.set_xticks(np.arange(n_metrics))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels, fontsize=_FS_TICK)
    ax.set_yticklabels(all_labels, fontsize=_FS_TICK)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')

    def _adaptive_text_color(value: float, fallback: str = 'black') -> str:
        if not np.isfinite(value):
            return fallback
        rgba = im.cmap(im.norm(value))
        luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        return 'white' if luminance < 0.52 else 'black'

    # --- Annotate cells ---
    for i in range(n_rows):
        for j in range(n_metrics):
            if has_obs and i == 0:
                # Absolute value for the obs2 baseline row
                val = obs_row[j]
                txt = f'{val:.2f}'
                txt_color = 'black'
            else:
                row_idx = i - (1 if has_obs else 0)
                val = ratio_data[row_idx, j] if has_obs else data[row_idx, j]
                txt = f'{val:.2f}\u00d7' if has_obs else f'{val:.2f}'
                txt_color = _adaptive_text_color(val)
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=8, color=txt_color, fontweight='bold', zorder=3)

    # --- Grid lines ---
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(n_metrics + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)

    return im

def plot_SIC_ts(sic_metric_obs, sic_metric_AllModel, model_labels, fig_name,
                line_style=None, color=None, model_spread_payloads=None, **kwargs):
    """Plot monthly climatological cycles of SIE, SIA, PIA, and MIZ.

    Four subplots arranged 2×2, each showing the 12-month seasonal cycle
    for all observation and model datasets.

    Args:
        sic_metric_obs:      List of obs metric dicts (from SIC_1M_metrics).
        sic_metric_AllModel: List of model metric dicts.
        model_labels:        Labels for [obs..., model...] datasets.
        fig_name:            Output file path.
        line_style, color:   Optional per-model style overrides.
    """
    # assemble data: obs + model → (N, 4, 12) array
    data1 = np.array([[d['SIE_clim'], d['SIA_clim'], d['PIA_clim'], d['MIZ_clim']]
                       for d in sic_metric_obs])
    data2 = np.array([[d['SIE_clim'], d['SIA_clim'], d['PIA_clim'], d['MIZ_clim']]
                       for d in sic_metric_AllModel])
    data = np.concatenate((data1, data2), axis=0)
    spread_data = None
    if isinstance(model_spread_payloads, (list, tuple)) and model_spread_payloads:
        try:
            spread_data = np.array([
                [
                    np.asarray((d or {}).get('SIE_clim', np.array([])), dtype=float),
                    np.asarray((d or {}).get('SIA_clim', np.array([])), dtype=float),
                    np.asarray((d or {}).get('PIA_clim', np.array([])), dtype=float),
                    np.asarray((d or {}).get('MIZ_clim', np.array([])), dtype=float),
                ]
                for d in model_spread_payloads
            ], dtype=object)
        except Exception:
            spread_data = None
    N = data.shape[0]
    n_obs = data1.shape[0]
    group_rank = {}
    for lb in model_labels:
        if _is_group_label(lb) and lb not in group_rank:
            group_rank[lb] = len(group_rank)

    # 2×2 fixed layout — use the standard (cols*5, rows*4.5) figsize
    fig, ax = plt.subplots(2, 2, figsize=_line_figsize(11, 9.8))
    ax = ax.flatten()

    title = ['(a) Sea Ice Extent', '(b) Sea Ice Area',
             '(c) Pack Ice Area', '(d) Marginal Ice Zone']

    for ii in range(4):
        for jj in range(N):
            # append Jan to close the annual cycle
            y = np.append(data[jj, ii, :], data[jj, ii, 0])
            label = model_labels[jj] if jj < len(model_labels) else f'dataset{jj + 1}'
            if jj < n_obs:
                style = _obs_style(jj)
                ax[ii].plot(y, label=label, **style)
            elif _is_group_label(label):
                style = _group_style(group_rank.get(label, 0))
                ax[ii].plot(y, label=label, **style)
            else:
                ls, cr = _get_style(jj, n_obs, line_style, color)
                style = {'linestyle': ls, 'color': cr, 'lw': _line_model_width()}
                ax[ii].plot(y, linestyle=ls, color=cr,
                            lw=_line_model_width(), label=label, **kwargs)
            if (jj >= n_obs) and (spread_data is not None):
                spread_idx = jj - n_obs
                if spread_idx < spread_data.shape[0]:
                    s_src = np.asarray(spread_data[spread_idx, ii], dtype=float).reshape(-1)
                    if s_src.size >= 1:
                        s = np.append(s_src, s_src[0])
                        n_use = min(y.size, s.size)
                        _plot_group_std_band(
                            ax[ii],
                            np.arange(n_use, dtype=float),
                            np.asarray(y[:n_use], dtype=float),
                            np.asarray(s[:n_use], dtype=float),
                            style,
                        )

        _apply_month_ticks(ax[ii], interval=2, rotation=30, use_datetime=False)
        ax[ii].tick_params(axis='y', labelsize=_FS_TICK)
        _apply_light_grid(ax[ii])
        ax[ii].set_ylabel(r'$10^6\ km^2$', fontsize=_FS_AXIS_LABEL)
        ax[ii].set_title(title[ii], fontsize=_FS_SUBPLOT_TITLE)
        ax[ii].set_xlim([-0.25, 12.25])
        ax[ii].set_ylim([0, 20])

    handles, labels_leg = ax[0].get_legend_handles_labels()
    _finalize_line_layout(fig, legend_location='none', pad=0.85)
    fig.subplots_adjust(
        hspace=_line_subplot_hspace(0.30),
        wspace=_line_subplot_wspace(0.34),
    )
    _place_adaptive_bottom_legend(
        fig, ax, handles, labels_leg,
        target_rows=3,
        fontsize=max(10, _FS_LEGEND),
        y_gap=0.020,
    )
    _save_fig(fig_name)

def plot_IIEE_ts(SIC_DIFF_AllModel, model_labels, fig_name,
                 line_style=None, color=None, obs_uncertainty=None,
                 obs_uncertainty_label='Obs uncertainty (obs2-obs1)',
                 obs_count=1, legend_y_gap=None, spread_payloads=None, **kwargs):
    """Plot monthly climatological cycles of IIEE and Over-/Under-estimation.

    Two subplots side by side: (a) Integrated Ice-Edge Error, (b) O - U.

    Args:
        SIC_DIFF_AllModel: List of model-vs-obs diff dicts (from SIC_2M_metrics).
        model_labels:      Labels for each model.
        fig_name:          Output file path.
        line_style, color: Optional per-model style overrides.
        obs_uncertainty:   Optional dict from obs2-vs-obs1 comparison to show observational uncertainty.
    """
    # assemble data: (N, 2, 12) — monthly climatologies of IIEE and O-U
    data = np.array([[d['IIEE_clim_diff'], d['O_U_clim_diff']]
                      for d in SIC_DIFF_AllModel])
    N = data.shape[0]
    group_rank = {}
    for lb in model_labels:
        if _is_group_label(lb) and lb not in group_rank:
            group_rank[lb] = len(group_rank)

    # 1×2 fixed layout — use the standard (cols*5, rows*4.5) figsize
    fig, ax = plt.subplots(1, 2, figsize=_line_figsize(10, 4.5))
    ax = ax.flatten()

    title = ['(a) Integrated Ice-Edge Error', '(b) Over. - Under.']
    for ii in range(2):
        # Plot observational uncertainty as shaded area or dashed line
        if obs_uncertainty is not None:
            if ii == 0:  # IIEE panel
                obs_iiee = np.append(obs_uncertainty['IIEE_clim_diff'],
                                     obs_uncertainty['IIEE_clim_diff'][0])
                ax[ii].fill_between(np.arange(13), 0, obs_iiee,
                                    alpha=0.2, color='gray',
                                    label=str(obs_uncertainty_label))
                ax[ii].plot(obs_iiee, linestyle='--', color='black',
                           lw=_line_model_width(1.0), alpha=0.7)
            else:  # O-U panel
                obs_ou = np.append(obs_uncertainty['O_U_clim_diff'],
                                   obs_uncertainty['O_U_clim_diff'][0])
                ax[ii].fill_between(np.arange(13), 0, obs_ou,
                                    alpha=0.2, color='gray',
                                    label=str(obs_uncertainty_label))
                ax[ii].plot(obs_ou, linestyle='--', color='black',
                           lw=_line_model_width(1.0), alpha=0.7)

        # Plot model-obs differences (use the same model style index mapping
        # as other SIconc line charts so each model keeps one style globally).
        obs_n = max(1, int(obs_count))
        for jj in range(N):
            y = np.append(data[jj, ii, :], data[jj, ii, 0])
            label = model_labels[jj] if jj < len(model_labels) else f'dataset{jj + 1}'
            if _is_group_label(label):
                style = _group_style(group_rank.get(label, 0))
                ax[ii].plot(y, label=label, **style)
            else:
                ls, cr = _get_style(jj + obs_n, obs_n, line_style, color)
                style = {'linestyle': ls, 'color': cr, 'lw': _line_model_width()}
                ax[ii].plot(y, linestyle=ls, color=cr,
                            lw=_line_model_width(), label=label, **kwargs)
            if isinstance(spread_payloads, (list, tuple)) and jj < len(spread_payloads):
                s_dict = spread_payloads[jj] or {}
                key_name = 'IIEE_clim_diff' if ii == 0 else 'O_U_clim_diff'
                s_src = np.asarray(s_dict.get(key_name, np.array([])), dtype=float).reshape(-1)
                if s_src.size >= 1:
                    s = np.append(s_src, s_src[0])
                    n_use = min(y.size, s.size)
                    _plot_group_std_band(
                        ax[ii],
                        np.arange(n_use, dtype=float),
                        np.asarray(y[:n_use], dtype=float),
                        np.asarray(s[:n_use], dtype=float),
                        style,
                    )

        # add zero line to the second subplot
        if ii == 1:
            ax[ii].plot(np.arange(-1, 14), np.zeros(15),
                        **_line_zero_style(color='gray', linewidth=0.6, alpha=0.5))

        _apply_month_ticks(ax[ii], interval=2, rotation=30, use_datetime=False)
        ax[ii].tick_params(axis='y', labelsize=_FS_TICK)
        _apply_light_grid(ax[ii])
        ax[ii].set_ylabel(r'$10^6\ km^2$', fontsize=_FS_AXIS_LABEL)
        ax[ii].set_title(title[ii], fontsize=_FS_SUBPLOT_TITLE)
        ax[ii].set_xlim([-0.25, 12.25])

    # Shared legend below all subplots to avoid overlapping data lines
    handles, labels_leg = [], []
    seen = set()
    for _ax in ax:
        h_i, l_i = _ax.get_legend_handles_labels()
        for hh, ll in zip(h_i, l_i):
            if ll in seen:
                continue
            seen.add(ll)
            handles.append(hh)
            labels_leg.append(ll)
    _finalize_line_layout(fig, legend_location='none', pad=0.82)
    fig.subplots_adjust(wspace=_line_subplot_wspace(0.30))
    _place_adaptive_bottom_legend(
        fig, ax, handles, labels_leg,
        target_rows=3,
        fontsize=max(9, _FS_LEGEND - 1),
        y_gap=float(0.040 if legend_y_gap is None else legend_y_gap),
    )
    _save_fig(fig_name)

def plot_SIC_ano(sic_metric_obs, sic_metric_AllModel, model_labels, year_range, fig_name,
                 line_style=None, color=None, hms=None, model_spread_payloads=None, **kwargs):
    """Plot monthly anomaly time series for SIE, SIA, PIA, and MIZ.

    Four vertically stacked subplots, each showing the deseasonalised anomaly
    with one shared figure-level legend.

    Args:
        sic_metric_obs:      List of obs metric dicts.
        sic_metric_AllModel: List of model metric dicts.
        model_labels:        Labels for [obs..., model...] datasets.
        year_range:          [start_year, end_year].
        fig_name:            Output file path.
        line_style, color:   Optional per-model style overrides.
        hms:                 Hemisphere (unused, kept for API compat).
    """
    # assemble anomalies, trend slopes, standard deviations, and p-values
    keys = ['SIE', 'SIA', 'PIA', 'MIZ']
    data1 = np.array([[d[f'{k}_ano'] for k in keys] for d in sic_metric_obs])
    data2 = np.array([[d[f'{k}_ano'] for k in keys] for d in sic_metric_AllModel])
    data = np.concatenate((data1, data2), axis=0)

    data1_slope = np.array([[_extract_trend_slope_pvalue(d.get(f'{k}_ano_tr'))[0] for k in keys] for d in sic_metric_obs])
    data2_slope = np.array([[_extract_trend_slope_pvalue(d.get(f'{k}_ano_tr'))[0] for k in keys] for d in sic_metric_AllModel])
    data_slope = np.concatenate((data1_slope * 12 * 10, data2_slope * 12 * 10), axis=0)

    data1_p = np.array([[_extract_trend_slope_pvalue(d.get(f'{k}_ano_tr'))[1] for k in keys] for d in sic_metric_obs])
    data2_p = np.array([[_extract_trend_slope_pvalue(d.get(f'{k}_ano_tr'))[1] for k in keys] for d in sic_metric_AllModel])
    data_p = np.concatenate((data1_p, data2_p), axis=0)

    metrics_all = list(sic_metric_obs) + list(sic_metric_AllModel)
    data_mean = np.array([
        [
            np.nanmean(np.asarray(
                d.get(f'{k}_ts', d.get(f'{k}_clim', np.array([np.nan]))),
                dtype=float,
            ))
            for k in keys
        ]
        for d in metrics_all
    ])

    N, _, _ = data.shape
    n_obs = data1.shape[0]
    group_rank = {}
    for lb in model_labels:
        if _is_group_label(lb) and lb not in group_rank:
            group_rank[lb] = len(group_rank)

    fig, ax = plt.subplots(4, 1, figsize=_line_figsize(12.4, 10.0))
    ax = ax.flatten()

    year_sta, year_end = year_range[0], year_range[1]
    dates = pd.date_range(f'{year_sta}-01-01',
                          periods=12 * (year_end - year_sta + 1), freq='ME')

    title = ['(a) SIE anomalies', '(b) SIA anomalies',
             '(c) PIA anomalies', '(d) MIZ anomalies']

    for ii in range(4):
        # zero line
        ax[ii].plot(dates, np.zeros(dates.shape),
                    **_line_zero_style(alpha=0.7))
        for jj in range(N):
            dataset_label = model_labels[jj] if jj < len(model_labels) else f'dataset{jj + 1}'
            label_str = dataset_label
            if jj < n_obs:
                style = _obs_style(jj)
                ax[ii].plot(dates, data[jj, ii, :], label=label_str, **style)
            elif _is_group_label(label_str):
                style = _group_style(group_rank.get(label_str, 0))
                ax[ii].plot(dates, data[jj, ii, :], label=label_str, **style)
            else:
                ls, cr = _get_style(jj, n_obs, line_style, color)
                style = {'color': cr, 'linestyle': ls, 'linewidth': _line_model_width()}
                ax[ii].plot(dates, data[jj, ii, :], linestyle=ls, color=cr,
                            lw=_line_model_width(), label=label_str, **kwargs)
            if jj >= n_obs and isinstance(model_spread_payloads, (list, tuple)):
                spread_idx = jj - n_obs
                if spread_idx < len(model_spread_payloads):
                    spread_dict = model_spread_payloads[spread_idx] or {}
                    s_vals = np.asarray(
                        spread_dict.get(f'{keys[ii]}_ano', np.array([])),
                        dtype=float,
                    ).reshape(-1)
                    if s_vals.size > 0:
                        y_vals = np.asarray(data[jj, ii, :], dtype=float).reshape(-1)
                        n_use = min(y_vals.size, s_vals.size, dates.size)
                        _plot_group_std_band(
                            ax[ii],
                            dates[:n_use],
                            y_vals[:n_use],
                            s_vals[:n_use],
                            style,
                        )

        # x-axis date format
        ax[ii].xaxis.set_major_locator(mdates.YearLocator())
        if ii < 3:
            ax[ii].set_xticklabels([''])
            ax[ii].tick_params(axis='x', bottom=False)
        else:
            ax[ii].xaxis.set_minor_locator(mdates.MonthLocator())
            ax[ii].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax[ii].tick_params(axis='x', rotation=45, labelsize=_FS_TICK)
        ax[ii].tick_params(axis='y', labelsize=_FS_TICK)
        ax[ii].set_ylabel(r'$10^6\ km^2$', fontsize=_FS_AXIS_LABEL)
        _apply_light_grid(ax[ii])
        ax[ii].text(
            -0.005, 1.03, title[ii],
            ha='left', va='bottom',
            transform=ax[ii].transAxes,
            fontsize=_FS_SUBPLOT_TITLE,
            clip_on=False,
        )

    handles, labels_leg = ax[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_leg,
        loc='center left', bbox_to_anchor=(0.80, 0.5),
        ncol=1, fontsize=max(11, _FS_LEGEND + 2), frameon=True,
        borderaxespad=0.0,
    )
    _finalize_line_layout(fig, legend_location='right', pad=0.78)
    fig.subplots_adjust(left=0.09, right=0.78, hspace=_line_subplot_hspace(0.34))
    _save_fig(fig_name)

def plot_SIC_map(grid_nc_file, data, model_labels, hms,
                 sic_range, diff_range, sic_cm='YlGnBu', diff_cm='RdBu_r',
                 unit='', sic_thre=None, plot_mode='mixed', fig_name=None, **kwargs):
    """Plot spatial maps of SIC fields.

    Args:
        grid_nc_file: Grid NetCDF with lon/lat variables.
        data:         (N, nx, ny) array — obs in data[0], others in data[1:].
        model_labels: Length-N list of labels.
        hms:          Hemisphere ('sh' or 'nh').
        sic_range:    [vmin, vmax] for raw-value panels.
        diff_range:   [vmin, vmax] for difference panels.
        sic_cm:       Colormap for raw-value panels.
        diff_cm:      Colormap for difference panels.
        unit:         Colorbar label.
        sic_thre:     Optional lower threshold to mask low-SIC cells in obs.
        plot_mode:    'mixed' (default, obs1 raw + diffs), 'raw' (all raw),
                      or 'diff' (all differences vs data[0]).
        fig_name:     Output file path.
    """
    with xr.open_dataset(grid_nc_file) as ds:
        lon, lat = np.array(ds['lon']), np.array(ds['lat'])

    N_total = data.shape[0]

    # Determine panels based on plot_mode
    if plot_mode == 'diff':
        if N_total < 2:
            return
        # First panel = obs1 raw, remaining panels = differences vs obs1
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

    proj = ccrs.Stereographic(
        central_latitude=-90 if hms == 'sh' else 90, central_longitude=0)
    _xy = proj.transform_points(ccrs.PlateCarree(), lon, lat)
    x_proj, y_proj = _xy[..., 0], _xy[..., 1]

    fig, ax = plt.subplots(rows, cols, figsize=_map_figsize(cols, rows),
                           subplot_kw={'projection': proj})
    ax = ax.flatten()

    cmp_raw = plt.get_cmap(sic_cm, 10)
    cmp_diff = plt.get_cmap(diff_cm, 21)

    im_raw = None
    im_diff = None

    for ii in range(N):
        polar_map(hms, ax[ii])
        field = np.copy(plot_data[ii, :, :])
        panel_label = plot_labels[ii] if ii < len(plot_labels) else f'dataset{ii + 1}'
        is_model_panel = not _is_obs_label(panel_label)

        if plot_mode == 'raw':
            # All panels: raw values with ice-edge contour
            if is_model_panel:
                field = _mask_model_zeros(field)
            if sic_thre is not None:
                field[field <= sic_thre] = np.nan
            mask = np.where(np.isnan(field), np.nan, (field >= 15).astype(float))
            im_raw = ax[ii].pcolormesh(lon, lat, field,
                                       vmin=sic_range[0], vmax=sic_range[1],
                                       transform=ccrs.PlateCarree(), cmap=cmp_raw)
            ax[ii].contour(
                x_proj, y_proj, mask, levels=[0.5],
                colors='c', linewidths=_map_contour_linewidth(1.0),
            )

        elif plot_mode == 'diff':
            if ii == 0:
                # First panel: obs1 raw values
                if sic_thre is not None:
                    field[field <= sic_thre] = np.nan
                mask = np.where(np.isnan(field), np.nan, (field >= 15).astype(float))
                im_raw = ax[ii].pcolormesh(lon, lat, field,
                                           vmin=sic_range[0], vmax=sic_range[1],
                                           transform=ccrs.PlateCarree(), cmap=cmp_raw)
                ax[ii].contour(
                    x_proj, y_proj, mask, levels=[0.5],
                    colors='c', linewidths=_map_contour_linewidth(1.0),
                )
            else:
                # Remaining panels: differences with dual ice-edge contours
                field = _mask_model_zeros(field)
                obs1 = data[0, :, :]
                sic_cur = data[ii, :, :]
                mask1 = np.where(np.isnan(obs1), np.nan, (obs1 >= 15).astype(float))
                mask2 = np.where(np.isnan(sic_cur), np.nan, (sic_cur >= 15).astype(float))
                im_diff = ax[ii].pcolormesh(lon, lat, field,
                                            vmin=diff_range[0], vmax=diff_range[1],
                                            transform=ccrs.PlateCarree(), cmap=cmp_diff)
                ax[ii].contour(
                    x_proj, y_proj, mask1, levels=[0.5],
                    colors='c', linewidths=_map_contour_linewidth(1.0),
                )
                ax[ii].contour(
                    x_proj, y_proj, mask2, levels=[0.5],
                    colors='y', linewidths=_map_contour_linewidth(1.0),
                )

        else:  # mixed (original behaviour)
            if ii == 0:
                if sic_thre is not None:
                    field[field <= sic_thre] = np.nan
                mask = np.where(np.isnan(field), np.nan, (field >= 15).astype(float))
                im_raw = ax[ii].pcolormesh(lon, lat, field,
                                           vmin=sic_range[0], vmax=sic_range[1],
                                           transform=ccrs.PlateCarree(), cmap=cmp_raw)
                ax[ii].contour(
                    x_proj, y_proj, mask, levels=[0.5],
                    colors='c', linewidths=_map_contour_linewidth(1.0),
                )
            else:
                sic0 = np.squeeze(data[0, :, :])
                mask1 = np.where(np.isnan(sic0), np.nan, (sic0 >= 15).astype(float))
                mask2 = np.where(np.isnan(field), np.nan, (field >= 15).astype(float))
                diff_f = field - data[0, :, :]
                diff_f = _mask_model_zeros(diff_f)
                im_diff = ax[ii].pcolormesh(lon, lat, diff_f,
                                            vmin=diff_range[0], vmax=diff_range[1],
                                            transform=ccrs.PlateCarree(), cmap=cmp_diff)
                ax[ii].contour(
                    x_proj, y_proj, mask1, levels=[0.5],
                    colors='c', linewidths=_map_contour_linewidth(1.0),
                )
                ax[ii].contour(
                    x_proj, y_proj, mask2, levels=[0.5],
                    colors='y', linewidths=_map_contour_linewidth(1.0),
                )

        ax[ii].set_title(panel_label, fontsize=max(22, _FS_SUBPLOT_TITLE + 1))

    for j in range(N, len(ax)):
        fig.delaxes(ax[j])

    _finalize_map_layout(fig, has_bottom_cbar=False)
    if plot_mode == 'raw':
        _bottom_cbar(fig, im_raw, [ax[i] for i in range(N)], label=unit)
    elif plot_mode == 'diff':
        # Diff-focused panel: keep one shared difference colorbar to avoid
        # overlapping the lower-row maps with a second raw colorbar.
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

def plot_trend_map(grid_nc_file, trend_data, pvalue_data, model_labels, hms,
                   trend_range, cmap='RdBu_r', unit='', plot_mode='raw',
                   fig_name=None, **kwargs):
    """Plot spatial trend maps with gray crosses for significant trends (p < 0.05).

    Args:
        grid_nc_file: Grid NetCDF with lon/lat variables.
        trend_data:   (N, nx, ny) array — trend values for each dataset.
        pvalue_data:  (N, nx, ny) array — p-values for each dataset.
        model_labels: Length-N list of labels.
        hms:          Hemisphere ('sh' or 'nh').
        trend_range:  [vmin, vmax] for all panels (symmetric around zero).
        cmap:         Colormap for trend values (default: 'RdBu_r').
        unit:         Colorbar label.
        plot_mode:    'raw' (default, all panels show raw trends) or
                      'diff' (N-1 panels show trend[i] - trend[0]).
        fig_name:     Output file path.
    """
    with xr.open_dataset(grid_nc_file) as ds:
        lon, lat = np.array(ds['lon']), np.array(ds['lat'])

    N_total = trend_data.shape[0]

    if plot_mode == 'diff':
        if N_total < 2:
            return
        # First panel = obs1 raw trend, remaining = trend differences
        diff_part = trend_data[1:] - trend_data[0:1]
        plot_trend = np.concatenate([trend_data[0:1], diff_part], axis=0)
        plot_pval = pvalue_data  # keep all p-values aligned
        plot_labels = list(model_labels)
        # Dynamic symmetric range for difference panels
        _abs_max = float(np.nanpercentile(np.abs(diff_part), 95))
        _abs_max = max(abs(trend_range[1]) * 0.5, _abs_max)
        diff_range = [-_abs_max, _abs_max]
        plot_range = trend_range  # raw range for obs1 panel
    else:
        plot_trend = trend_data
        plot_pval = pvalue_data
        plot_labels = list(model_labels)
        plot_range = trend_range

    N = plot_trend.shape[0]
    if N == 0:
        return
    nx = plot_trend.shape[1]
    rows, cols = _adaptive_grid(N)

    proj = ccrs.Stereographic(
        central_latitude=-90 if hms == 'sh' else 90, central_longitude=0)
    _xy = proj.transform_points(ccrs.PlateCarree(), lon, lat)
    x_proj, y_proj = _xy[..., 0], _xy[..., 1]

    fig, ax = plt.subplots(rows, cols, figsize=_map_figsize(cols, rows),
                           subplot_kw={'projection': proj})
    ax = ax.flatten()

    cmp = plt.get_cmap(cmap, 21)
    im_raw = None
    im_diff = None

    for ii in range(N):
        polar_map(hms, ax[ii])
        trend = np.copy(plot_trend[ii, :, :])
        pval = np.copy(plot_pval[ii, :, :])
        panel_label = plot_labels[ii] if ii < len(plot_labels) else f'dataset{ii + 1}'
        is_model_panel = not _is_obs_label(panel_label)

        if (plot_mode == 'raw') and is_model_panel:
            trend = _mask_model_zeros(trend)

        if plot_mode == 'diff' and ii == 0:
            # First panel: obs1 raw trend
            im_raw = ax[ii].pcolormesh(lon, lat, trend,
                                       vmin=plot_range[0], vmax=plot_range[1],
                                       transform=ccrs.PlateCarree(), cmap=cmp)
        elif plot_mode == 'diff':
            # Remaining panels: trend differences
            im_diff = ax[ii].pcolormesh(lon, lat, trend,
                                        vmin=diff_range[0], vmax=diff_range[1],
                                        transform=ccrs.PlateCarree(), cmap=cmp)
        else:
            im_raw = ax[ii].pcolormesh(lon, lat, trend,
                                       vmin=plot_range[0], vmax=plot_range[1],
                                       transform=ccrs.PlateCarree(), cmap=cmp)

        # Gray crosses for significant trends (p < 0.05)
        sig_mask = (pval < 0.05) & np.isfinite(trend)
        if np.any(sig_mask):
            step = max(1, int(nx / 36))
            lon_sub = lon[::step, ::step]
            lat_sub = lat[::step, ::step]
            sig_sub = sig_mask[::step, ::step]
            lon_sig = lon_sub[sig_sub]
            lat_sig = lat_sub[sig_sub]
            if len(lon_sig) > 0:
                ax[ii].scatter(
                    lon_sig, lat_sig,
                    s=12, c='0.45', alpha=0.7,
                    transform=ccrs.PlateCarree(),
                    marker='x', linewidths=0.45,
                )

        ax[ii].set_title(panel_label, fontsize=max(22, _FS_SUBPLOT_TITLE + 1))

    for j in range(N, len(ax)):
        fig.delaxes(ax[j])

    _finalize_map_layout(fig, has_bottom_cbar=False)
    if plot_mode == 'diff':
        # Diff-focused panel: keep one shared difference colorbar.
        if im_diff is not None:
            _bottom_cbar(fig, im_diff, [ax[i] for i in range(N)],
                         label=f'\u0394 {unit}', extend='auto')
        elif im_raw is not None:
            _bottom_cbar(fig, im_raw, [ax[0]], label=unit, extend='auto')
    else:
        if im_raw is not None:
            _bottom_cbar(fig, im_raw, [ax[i] for i in range(N)],
                         label=unit, extend='auto')

    _save_fig(fig_name)


def plot_sic_region_map(grid_nc_file: str, hms: str, fig_name: str = None):
    """Plot SITool sea-ice sector partition map for one hemisphere.

    The plotted sectors follow the same grouped region definitions used by
    ``utils.region_index`` and the regional scalar-table statistics.
    """
    hms_key = str(hms or '').lower()
    with xr.open_dataset(grid_nc_file) as ds:
        lon = np.array(ds['lon'])
        lat = np.array(ds['lat'])

    sectors = utils.get_hemisphere_sectors(hms_key, include_all=False)
    if not sectors:
        return

    grouped_region = np.full(lon.shape, np.nan, dtype=float)
    for idx, sector in enumerate(sectors, start=1):
        mask = utils.region_index(grid_file=grid_nc_file, hms=hms_key, sector=sector)
        grouped_region[mask] = float(idx)

    proj = ccrs.Stereographic(
        central_latitude=-90 if hms_key == 'sh' else 90,
        central_longitude=0,
    )
    fig, ax = plt.subplots(1, 1, figsize=_map_abs_figsize(7.8, 9.2), subplot_kw={'projection': proj})
    polar_map(hms_key, ax)

    # Use a discrete palette and override specific sector colors when needed.
    # In NH, make Hudson/Baffin Bays (HBB) clearly distinct from land grey.
    cmap_colors = list(plt.get_cmap('tab20', len(sectors))(np.arange(len(sectors))))
    if hms_key == 'nh':
        nh_color_overrides = {
            'HBB': '#d62728',  # vivid red for high contrast against land background
        }
        for ii, sector in enumerate(sectors):
            override = nh_color_overrides.get(sector)
            if override:
                cmap_colors[ii] = matplotlib.colors.to_rgba(override)
    cmap = matplotlib.colors.ListedColormap(cmap_colors, name=f'sea_ice_sector_{hms_key}')

    im = ax.pcolormesh(
        lon, lat, grouped_region,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=0.5, vmax=len(sectors) + 0.5,
    )
    ax.set_title(
        f'Sea-Ice Region Partition ({hms_key.upper()})',
        fontsize=_FS_SUBPLOT_TITLE,
    )

    tick_values = np.arange(1, len(sectors) + 1)
    tick_labels = [utils.get_sector_label(hms_key, s) for s in sectors]
    _finalize_map_layout(fig, has_bottom_cbar=False)
    cbar = fig.colorbar(
        im, ax=ax, orientation='horizontal',
        fraction=0.04, pad=0.08, ticks=tick_values,
        extend=_resolve_colorbar_extend(im, extend='auto'),
    )
    base_tick_fs = _MAP_CBAR_TICK_FONTSIZE if _MAP_CBAR_TICK_FONTSIZE is not None else max(11, _FS_CBAR_TICK - 1)
    base_label_fs = _MAP_CBAR_LABEL_FONTSIZE if _MAP_CBAR_LABEL_FONTSIZE is not None else max(9, _FS_CBAR_LABEL - 2)
    tick_fs = max(5, float(base_tick_fs) * 0.5)
    label_fs = max(4, float(base_label_fs) * 0.5)
    cbar.ax.set_xticklabels(tick_labels, rotation=25, ha='right', fontsize=tick_fs)
    cbar.ax.tick_params(labelsize=tick_fs)
    cbar.set_label('Sectors used by regional scalar statistics', fontsize=label_fs)

    _save_fig(fig_name)

__all__ = [
    "plot_siconc_key_month_ano",
    "plot_heat_map",
    "plot_SIC_ts",
    "plot_IIEE_ts",
    "plot_SIC_ano",
    "plot_SIC_map",
    "plot_trend_map",
    "plot_sic_region_map",
]
