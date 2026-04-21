# -*- coding: utf-8 -*-
"""Plotting routines split by diagnostic family."""

from scripts.plot_figs import core as _core
from scipy import ndimage

# Reuse shared plotting namespace and helpers from core module.
globals().update({k: v for k, v in _core.__dict__.items() if k not in globals()})


def _seasonal_ridging_mask(ds: xr.Dataset, season_idx: int) -> Optional[np.ndarray]:
    """Return seasonal ridging mask (0/1) for one season index."""
    if 'ridging_mask' in ds:
        ridging = np.array(ds['ridging_mask'])[season_idx, :, :]
        return np.asarray(ridging, dtype=float)

    if all(v in ds.variables for v in ('sic_mean', 'res_mean', 'div_mean')):
        sic_mean = np.array(ds['sic_mean'])[season_idx, :, :]
        res_mean = np.array(ds['res_mean'])[season_idx, :, :]
        div_mean = np.array(ds['div_mean'])[season_idx, :, :]
        valid = np.isfinite(sic_mean) & np.isfinite(res_mean) & np.isfinite(div_mean)
        ridging = np.where(
            valid,
            ((sic_mean > 90.0) & (res_mean < 0.0) & (div_mean > 0.0)).astype(float),
            np.nan,
        )
        return np.asarray(ridging, dtype=float)

    return None


def _resolve_lon_lat(
    ds: xr.Dataset,
    fallback: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve lon/lat arrays from a dataset with optional fallback."""
    lon_candidates = ('lon', 'longitude', 'nav_lon', 'LON', 'LONGITUDE')
    lat_candidates = ('lat', 'latitude', 'nav_lat', 'LAT', 'LATITUDE')

    lon_da = None
    lat_da = None
    for name in lon_candidates:
        if name in ds.variables:
            lon_da = ds[name]
            break
    for name in lat_candidates:
        if name in ds.variables:
            lat_da = ds[name]
            break

    if lon_da is None or lat_da is None:
        if fallback is not None:
            return (
                np.asarray(fallback[0], dtype=float),
                np.asarray(fallback[1], dtype=float),
            )
        raise KeyError('lon')

    return np.asarray(lon_da, dtype=float), np.asarray(lat_da, dtype=float)


def _plot_ridging_contour(ax, lon, lat, ridging_mask,
                          contour_color: str = 'k', contour_lw: float = 0.8) -> None:
    """Overlay ridging-region contour on one map axes."""
    if ridging_mask is None:
        return
    ridging = np.asarray(ridging_mask, dtype=float)
    valid = np.isfinite(ridging)
    if (not np.any(valid)) or (np.nanmax(ridging) < 0.5):
        return
    ridging_bin = valid & (ridging > 0.5)
    if not np.any(ridging_bin):
        return
    lw = _map_contour_linewidth(contour_lw)

    # Clean ridging mask for visualization to avoid noisy interleaved loops:
    # 1) close small gaps and fill tiny holes
    # 2) drop very small isolated components
    structure = np.ones((3, 3), dtype=bool)
    ridging_bin = ndimage.binary_closing(ridging_bin, structure=structure)
    ridging_bin = ndimage.binary_fill_holes(ridging_bin)
    labels, n_labels = ndimage.label(ridging_bin, structure=structure)
    if n_labels > 0:
        sizes = np.bincount(labels.ravel())
        min_cells = max(20, int(0.005 * ridging_bin.size))
        keep = np.zeros((n_labels + 1,), dtype=bool)
        keep_idx = np.where(sizes >= min_cells)[0]
        keep[keep_idx] = True
        keep[0] = False
        ridging_bin = keep[labels]
        if not np.any(ridging_bin):
            # If all components were filtered out, keep the largest one.
            largest_idx = int(np.argmax(sizes[1:]) + 1) if len(sizes) > 1 else 0
            if largest_idx > 0:
                ridging_bin = labels == largest_idx
    if not np.any(ridging_bin):
        return

    ridging_filled = ridging_bin.astype(float)
    try:
        # Build contour segments in map-projection coordinates first.
        # This avoids longitude seam (e.g. +180/-180) artifacts that can
        # generate visually incorrect self-connected loops on polar maps.
        lon_arr = np.asarray(lon, dtype=float)
        lat_arr = np.asarray(lat, dtype=float)
        pts = ax.projection.transform_points(ccrs.PlateCarree(), lon_arr, lat_arr)
        xproj = np.asarray(pts[..., 0], dtype=float)
        yproj = np.asarray(pts[..., 1], dtype=float)
        finite_xy = np.isfinite(xproj) & np.isfinite(yproj)
        zproj = np.where(finite_xy, np.asarray(ridging_filled, dtype=float), np.nan)
        if not np.any(np.isfinite(zproj)):
            return

        # Build contour segments off-screen, then draw as plain lines
        # directly in projection-space data coordinates on GeoAxes.
        fig_tmp, ax_tmp = plt.subplots(1, 1, figsize=(1, 1))
        try:
            cs = ax_tmp.contour(
                xproj,
                yproj,
                zproj,
                levels=[0.5],
            )
            segs = cs.allsegs[0] if getattr(cs, 'allsegs', None) else []
        finally:
            plt.close(fig_tmp)

        for seg in segs:
            arr = np.asarray(seg, dtype=float)
            if arr.ndim != 2 or arr.shape[0] < 2:
                continue
            if arr.shape[1] < 2:
                continue
            xy_ok = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
            if np.count_nonzero(xy_ok) < 2:
                continue
            arr2 = arr[xy_ok]
            ax.plot(
                arr2[:, 0], arr2[:, 1],
                color=contour_color,
                linewidth=lw,
            )
    except Exception:
        # Silently skip contour overlay if contour extraction/drawing fails.
        return


def _sicb_monthly_term_series(mon_file: str, term: str) -> Optional[np.ndarray]:
    """Return 13-point monthly cycle for one SICB term."""
    try:
        with _open_nc_ref(mon_file) as ds:
            if term not in ds.variables:
                return None
            vals = np.asarray(ds[term], dtype=float)
            if vals.ndim == 3:
                vals = np.nanmean(vals, axis=(1, 2))
            vals = np.asarray(vals, dtype=float).reshape(-1)
            if vals.size <= 0:
                return None
            return np.append(vals, vals[0])
    except Exception:
        return None


def _sicb_ridging_monthly_series(daily_file: str) -> tuple[np.ndarray, np.ndarray]:
    """Return monthly ridging-ratio time series as (x_datetime, y)."""
    with _open_nc_ref(daily_file) as ds:
        if 'ridging_ratio' in ds:
            ratio_daily = np.array(ds['ridging_ratio'], dtype=float).squeeze()
        elif all(v in ds.variables for v in ('sic', 'res', 'div')):
            sic = np.array(ds['sic'], dtype=float)
            res = np.array(ds['res'], dtype=float)
            div = np.array(ds['div'], dtype=float)
            ice_mask = np.isfinite(sic) & (sic >= 15.0)
            ridging_mask = ice_mask & (sic > 90.0) & np.isfinite(res) & np.isfinite(div) & (res < 0.0) & (div > 0.0)
            ice_count = np.sum(ice_mask, axis=(1, 2), dtype=float)
            ridging_count = np.sum(ridging_mask, axis=(1, 2), dtype=float)
            ratio_daily = np.where(ice_count > 0.0, ridging_count / ice_count, np.nan)
        else:
            return np.array([]), np.array([])

        time_idx = pd.to_datetime(np.array(ds['time'].values))
        if ratio_daily.ndim != 1 or ratio_daily.size != time_idx.size:
            return np.array([]), np.array([])
        valid = np.isfinite(ratio_daily)
        if not np.any(valid):
            return np.array([]), np.array([])
        series = pd.Series(ratio_daily[valid], index=time_idx[valid]).sort_index()
        monthly = series.resample('MS').mean()
        monthly = monthly[np.isfinite(monthly.values)]
        if monthly.empty:
            return np.array([]), np.array([])
        return np.asarray(monthly.index.to_pydatetime()), np.asarray(monthly.values, dtype=float)


def plot_SICB(filenm, hms,
              budget_term_str=None,
              unit=r'$\mathrm{\mathbf{\%/season}}$',
              vmin=-100, vmax=100, cbtick_bin=20, cm='RdBu_r',
              fig_name='',
              overlay_ridging=True,
              ridging_contour_color='k'):
    """Plot a 4×4 panel of SICB spatial patterns (seasons × budget terms).

    Each row is a season (DJF/MAM/JJA/SON), each column is a budget term
    (dadt, adv, div, res).  A shared horizontal colorbar is placed below.

    Args:
        filenm:          NetCDF file containing seasonal budget fields.
        hms:             Hemisphere ('sh' or 'nh').
        budget_term_str: List of variable names in *filenm* (default: dadt/adv/div/res).
        unit:            Colorbar label string.
        vmin, vmax:      Colour range limits.
        cbtick_bin:      Colorbar tick spacing.
        cm:              Matplotlib colormap name.
        fig_name:        Output file path.
    """
    if budget_term_str is None:
        budget_term_str = ['dadt', 'adv', 'div', 'res']

    cbtick = np.arange(float(vmin), float(vmax) + 0.5 * float(cbtick_bin), float(cbtick_bin), dtype=float)
    if cbtick.size < 2:
        cbtick = np.array([float(vmin), float(vmax)], dtype=float)
    cb_bounds = np.concatenate((
        [cbtick[0] - 0.5 * float(cbtick_bin)],
        0.5 * (cbtick[:-1] + cbtick[1:]),
        [cbtick[-1] + 0.5 * float(cbtick_bin)],
    ))
    cb_norm = matplotlib.colors.BoundaryNorm(cb_bounds, ncolors=plt.get_cmap(cm).N, clip=True)

    # season labels (differ between hemispheres)
    if hms == 'sh':
        ss_str = ['Summer (DJF)', 'Autumn (MAM)', 'Winter (JJA)', 'Spring (SON)']
    else:
        ss_str = ['Winter (DJF)', 'Spring (MAM)', 'Summer (JJA)', 'Autumn (SON)']

    proj = ccrs.Stereographic(
        central_latitude=-90 if hms == 'sh' else 90, central_longitude=0)
    fig, ax = plt.subplots(
        len(ss_str), len(budget_term_str), figsize=_map_abs_figsize(12, 13),
        subplot_kw={'projection': proj}, constrained_layout=True)

    # read NetCDF and draw each panel
    panel_vals: List[np.ndarray] = []
    with _open_nc_ref(filenm) as ds:
        lon, lat = _resolve_lon_lat(ds)
        for ii in range(len(ss_str)):
            ridging_mask = _seasonal_ridging_mask(ds, ii) if overlay_ridging else None
            for jj in range(len(budget_term_str)):
                data_p = np.array(ds[budget_term_str[jj]])[ii, :, :]
                data_p = _mask_model_zeros(data_p)
                finite = np.asarray(data_p, dtype=float)
                finite = finite[np.isfinite(finite)]
                if finite.size > 0:
                    panel_vals.append(finite)
                im = polar_map(hms, ax[ii, jj]).pcolormesh(
                    lon, lat, data_p,
                    norm=cb_norm,
                    cmap=plt.get_cmap(cm), transform=ccrs.PlateCarree())
                if budget_term_str[jj] == 'res':
                    _plot_ridging_contour(
                        ax[ii, jj], lon, lat, ridging_mask,
                        contour_color=ridging_contour_color,
                    )
                if ii == 0:
                    ax[ii, jj].set_title(budget_term_str[jj], fontsize=20, weight='bold')
                if jj == 0:
                    ax[ii, jj].set_ylabel(ss_str[ii], fontsize=20, weight='bold')

    # shared horizontal colorbar
    all_data = np.concatenate(panel_vals) if panel_vals else np.array([], dtype=float)
    cb = fig.colorbar(im, ax=ax.ravel().tolist(), orientation='horizontal',
                      fraction=0.03, pad=0.04,
                      boundaries=cb_bounds,
                      ticks=cbtick,
                      spacing='proportional',
                      extend=_resolve_colorbar_extend(
                          im, data=all_data, vmin=vmin, vmax=vmax, extend='auto'
                      ))
    cb.ax.set_xticklabels([f'{v:g}' for v in cbtick], weight='bold')
    cb.ax.tick_params(labelsize=max(16, _FS_CBAR_TICK))
    cb.ax.set_title(unit, fontsize=14, loc='center', weight='bold')

    _save_fig(fig_name)

def plot_SICB_ts(mon_clim_files, labels, hms, fig_name=None,
                 line_style=None, color=None, n_obs=None,
                 group_member_files=None):
    """Plot monthly climatological cycles of SICB budget terms from MonClim NetCDF files.

    Reads the four budget terms (dadt, adv, div, res) from each dataset's
    MonClim file and plots them as seasonal cycles in a 2×2 panel layout.

    Args:
        mon_clim_files: List of paths to *_MonClim.nc files (obs first, then models).
        labels:         Length-N list of dataset labels.
        hms:            Hemisphere ('sh' or 'nh').
        fig_name:       Output file path.
    """
    terms = ['dadt', 'adv', 'div', 'res']
    term_titles = ['(a) Net change (dadt)', '(b) Advection (adv)',
                   '(c) Divergence (div)', '(d) Residual (res)']
    N = len(mon_clim_files)
    label_seq = list(labels) if labels is not None else []
    group_rank: Dict[str, int] = {}
    for lb in label_seq:
        if _is_group_label(lb) and lb not in group_rank:
            group_rank[lb] = len(group_rank)
    if n_obs is None:
        n_obs = 0
        for lbl in label_seq[:N]:
            if _is_obs_label(lbl):
                n_obs += 1
            else:
                break
    n_obs = max(0, min(int(n_obs), N))

    # 2×2 fixed layout — use the standard (cols*5, rows*4.5) figsize
    fig, ax = plt.subplots(2, 2, figsize=_line_figsize(11.2, 9.8))
    ax = ax.flatten()

    for jj, term in enumerate(terms):
        ax[jj].axhline(0, **_line_zero_style(alpha=0.5))
        for ii, fpath in enumerate(mon_clim_files):
            y = _sicb_monthly_term_series(fpath, term)
            if y is None:
                continue

            if ii < n_obs:
                style = _obs_style(ii)
                ax[jj].plot(y, label=labels[ii], **style)
            elif _is_group_label(labels[ii]):
                style = _group_style(group_rank.get(labels[ii], 0))
                ax[jj].plot(y, label=labels[ii], **style)
            else:
                ls, cr = _get_style(ii, n_obs, line_style, color)
                style = {'linestyle': ls, 'color': cr, 'lw': _line_model_width()}
                ax[jj].plot(y, linestyle=ls, color=cr, lw=_line_model_width(), label=labels[ii])

            if (
                isinstance(group_member_files, dict)
                and _is_group_label(labels[ii])
                and labels[ii] in group_member_files
            ):
                member_paths = [p for p in group_member_files.get(labels[ii], []) if p]
                member_cycles = []
                for mp in member_paths:
                    ys = _sicb_monthly_term_series(mp, term)
                    if ys is not None:
                        member_cycles.append(np.asarray(ys, dtype=float))
                if member_cycles:
                    arr = np.asarray(member_cycles, dtype=float)
                    if arr.ndim == 2 and arr.shape[1] >= 1:
                        std = np.nanstd(arr, axis=0)
                        n_use = min(len(y), len(std))
                        _plot_group_std_band(
                            ax[jj],
                            np.arange(n_use, dtype=float),
                            np.asarray(y[:n_use], dtype=float),
                            np.asarray(std[:n_use], dtype=float),
                            style,
                        )

        ax[jj].set_title(term_titles[jj], fontsize=_FS_SUBPLOT_TITLE)
        _apply_month_ticks(ax[jj], interval=2, rotation=30, use_datetime=False)
        ax[jj].tick_params(axis='y', labelsize=_FS_TICK)
        ax[jj].set_ylabel(r'%/month', fontsize=_FS_AXIS_LABEL)
        _apply_light_grid(ax[jj])
        ax[jj].set_xlim([-0.25, 12.25])

    # Shared legend below all subplots to avoid overlapping data lines
    _finalize_line_layout(fig, legend_location='bottom', pad=0.86)
    fig.subplots_adjust(
        hspace=_line_subplot_hspace(0.34),
        wspace=_line_subplot_wspace(0.30),
        bottom=_line_bottom_margin(0.16),
    )

    handles, labels_leg = ax[0].get_legend_handles_labels()
    if handles:
        positions = [aa.get_position() for aa in ax]
        x0 = min(pos.x0 for pos in positions)
        x1 = max(pos.x1 for pos in positions)
        legend_width = max(0.15, x1 - x0)
        fig.legend(
            handles, labels_leg,
            loc='lower left',
            bbox_to_anchor=(x0, _line_bottom_legend_y(0.035), legend_width, 0.05),
            bbox_transform=fig.transFigure,
            mode='expand',
            ncol=max(1, min(5, len(labels_leg))),
            fontsize=_FS_LEGEND,
            frameon=True,
            borderaxespad=0.0,
            columnspacing=1.0,
            handlelength=2.0,
        )
    _save_fig(fig_name)


def plot_SICB_ridging_ts(daily_files, labels, hms, fig_name=None,
                         line_style=None, color=None, n_obs=None,
                         group_member_files=None):
    """Plot monthly mean ridging-grid ratio time series from daily SICB files."""
    N = len(daily_files)
    label_seq = list(labels) if labels is not None else []
    group_rank: Dict[str, int] = {}
    for lb in label_seq:
        if _is_group_label(lb) and lb not in group_rank:
            group_rank[lb] = len(group_rank)
    if n_obs is None:
        n_obs = 0
        for lbl in label_seq[:N]:
            if _is_obs_label(lbl):
                n_obs += 1
            else:
                break
    n_obs = max(0, min(int(n_obs), N))

    fig, ax = plt.subplots(1, 1, figsize=_line_figsize(12, 4.4))
    ax.axhline(0, **_line_zero_style(alpha=0.5))

    def _collect_member_series(member_paths):
        """Load ridging monthly series from group member daily files."""
        out = []
        for mp in (member_paths or []):
            if not (isinstance(mp, str) and Path(mp).exists()):
                continue
            try:
                x_m, y_m = _sicb_ridging_monthly_series(mp)
            except Exception:
                continue
            y_arr = np.asarray(y_m, dtype=float).reshape(-1)
            if y_arr.size <= 0:
                continue
            out.append((pd.to_datetime(np.asarray(x_m).reshape(-1)), y_arr))
        return out

    def _align_member_series(member_series, target_x=None):
        """Align member series to one monthly axis and return mean/std-ready array."""
        if not member_series:
            return None

        if target_x is None:
            month_set = set()
            for xs, _ in member_series:
                for ts in pd.to_datetime(xs):
                    month_set.add(pd.Timestamp(ts).to_period('M'))
            if not month_set:
                return None
            month_axis = sorted(month_set)
            x_axis = pd.to_datetime([mm.to_timestamp(how='start') for mm in month_axis])
        else:
            x_axis = pd.to_datetime(np.asarray(target_x).reshape(-1))
            if x_axis.size <= 0:
                return None

        month_to_idx = {pd.Timestamp(ts).to_period('M'): ii for ii, ts in enumerate(x_axis)}
        aligned = []
        for xs, ys in member_series:
            arr = np.full(x_axis.size, np.nan, dtype=float)
            for ts, val in zip(pd.to_datetime(xs), np.asarray(ys, dtype=float)):
                pos = month_to_idx.get(pd.Timestamp(ts).to_period('M'))
                if pos is not None and np.isfinite(val):
                    arr[pos] = float(val)
            aligned.append(arr)
        if not aligned:
            return None

        mat = np.asarray(aligned, dtype=float)
        mean_vals = np.nanmean(mat, axis=0)
        if not np.any(np.isfinite(mean_vals)):
            return None
        return x_axis, mean_vals, mat

    legend_handles = []
    legend_labels = []

    for ii, fpath in enumerate(daily_files):
        label_i = labels[ii] if ii < len(labels) else f'dataset{ii + 1}'
        is_group = _is_group_label(label_i)

        x = None
        y = None
        member_aligned = None

        # 1) Prefer direct precomputed group/model daily file if it exists.
        if isinstance(fpath, str) and Path(fpath).exists():
            try:
                x_raw, y_raw = _sicb_ridging_monthly_series(fpath)
                y_arr = np.asarray(y_raw, dtype=float).reshape(-1)
                if y_arr.size > 0:
                    x = pd.to_datetime(np.asarray(x_raw).reshape(-1))
                    y = y_arr
            except Exception:
                pass

        # 2) Group fallback: if direct group file is unavailable, compute
        #    group mean series from member daily files and plot it directly.
        if is_group and isinstance(group_member_files, dict):
            member_series = _collect_member_series(group_member_files.get(label_i, []))
            if member_series:
                aligned = _align_member_series(member_series, target_x=x)
                if aligned is not None:
                    x_grp, y_grp, member_mat = aligned
                    if x is None or y is None or np.asarray(y).size <= 0:
                        x = x_grp
                        y = y_grp
                    member_aligned = member_mat

        if x is None or y is None or np.asarray(y).size <= 0:
            continue

        if ii < n_obs:
            style = _obs_style(ii)
            line_handle = ax.plot(x, y, label=label_i, **style)[0]
        elif is_group:
            style = _group_style(group_rank.get(label_i, 0))
            line_handle = ax.plot(x, y, label=label_i, **style)[0]
        else:
            ls, cr = _get_style(ii, n_obs, line_style, color)
            style = {'linestyle': ls, 'color': cr, 'lw': _line_model_width()}
            line_handle = ax.plot(x, y, linestyle=ls, color=cr, lw=_line_model_width(), label=label_i)[0]

        legend_handles.append(line_handle)
        legend_labels.append(str(label_i))

        if (
            is_group
            and member_aligned is not None
        ):
            std = np.nanstd(member_aligned, axis=0)
            n_use = min(len(y), len(std))
            _plot_group_std_band(
                ax,
                np.asarray(x[:n_use]),
                np.asarray(y[:n_use], dtype=float),
                np.asarray(std[:n_use], dtype=float),
                style,
            )

    ax.set_title('Monthly mean ridging-grid ratio', fontsize=_FS_SUBPLOT_TITLE)
    ax.set_ylabel('Ridging ratio', fontsize=_FS_AXIS_LABEL)
    _apply_date_ticks(ax, minticks=5, maxticks=10)
    _apply_light_grid(ax)
    ax.tick_params(axis='x', labelsize=_FS_TICK, rotation=20)
    ax.tick_params(axis='y', labelsize=_FS_TICK)
    _finalize_line_layout(fig, legend_location='bottom', pad=0.92)
    if legend_handles:
        dedup_handles = []
        dedup_labels = []
        seen_labels = set()
        for hh, ll in zip(legend_handles, legend_labels):
            key = str(ll)
            if key in seen_labels:
                continue
            seen_labels.add(key)
            dedup_handles.append(hh)
            dedup_labels.append(key)

        ax_pos = ax.get_position()
        legend_center_x = 0.5 * (ax_pos.x0 + ax_pos.x1)
        fig.legend(
            dedup_handles, dedup_labels,
            loc='lower center', bbox_to_anchor=(legend_center_x, _line_bottom_legend_y(-0.01)),
            ncol=max(1, min(5, len(dedup_labels))),
            fontsize=_FS_LEGEND, frameon=True,
        )
    _save_fig(fig_name)


def plot_SICB2(file_list, model_labels, hms, unit=r'$\mathrm{\%/season}$',
               vmin=-100, vmax=100, cbtick_bin=20, cm='RdBu_r', fig_label=None,
               overlay_ridging=True, ridging_contour_color='k'):
    """

    :return:
    """
    N = len(file_list)

    if hms == 'sh':
        ss_str = ['Summer (DJF)', 'Autumn (MAM)', 'Winter (JJA)', 'Spring (SON)']
    elif hms == 'nh':
        ss_str = ['Winter (DJF)', 'Spring (MAM)', 'Summer (JJA)', 'Autumn (SON)']

    budget_term_str = ['dadt', 'adv', 'div', 'res']
    cbtick = np.arange(vmin, vmax + 1, cbtick_bin)

    ref_lon = None
    ref_lat = None
    for file_name in file_list:
        try:
            with _open_nc_ref(file_name) as ds_ref:
                ref_lon, ref_lat = _resolve_lon_lat(ds_ref)
                break
        except Exception:
            continue
    if ref_lon is None or ref_lat is None:
        raise KeyError('lon')
    fallback_coords = (ref_lon, ref_lat)

    for season in range(4):
        print(ss_str[season])
        rows, cols = N, 4

        proj = ccrs.Stereographic(central_latitude=-90 if hms == 'sh' else 90, central_longitude=0)

        fig, ax = plt.subplots(
            rows, cols, figsize=_map_abs_figsize(cols * 2.95, rows * 3.35), subplot_kw={'projection': proj}
        )
        if rows == 1:
            ax = np.array([ax])
        for ii in range(N):
            with _open_nc_ref(file_list[ii]) as ds:
                lon, lat = _resolve_lon_lat(ds, fallback=fallback_coords)
                ridging_mask = _seasonal_ridging_mask(ds, season) if overlay_ridging else None
                for jj in range(4):
                    data_p = np.array(ds[budget_term_str[jj]])[season, :, :]
                    data_p = _mask_model_zeros(data_p)

                    im1 = polar_map(hms, ax[ii, jj]).pcolormesh(lon, lat, data_p, vmin=vmin, vmax=vmax, cmap=plt.get_cmap(cm), transform=ccrs.PlateCarree())
                    if budget_term_str[jj] == 'res':
                        _plot_ridging_contour(
                            ax[ii, jj], lon, lat, ridging_mask,
                            contour_color=ridging_contour_color,
                        )

                    if ii == 0:
                        ax[ii, jj].set_title(budget_term_str[jj], fontsize=20, weight='bold')
                    if jj == 0:
                        row_label = str(model_labels[ii])
                        row_label = row_label.replace('+', '\n+\n')
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

        fig_name_out = f'{fig_label}_{ss_str[season]}.png'
        _finalize_map_layout(fig, has_bottom_cbar=False, pad=0.70)
        fig.subplots_adjust(
            wspace=max(0.008, float(_MAP_SUBPLOT_WSPACE) * 0.40),
            hspace=max(0.06, float(_MAP_SUBPLOT_HSPACE) * 0.85),
        )
        # Colorbar at the bottom to avoid overlapping subplots
        _bottom_cbar(fig, im1, ax.ravel().tolist(), label=unit, pad=0.016, fraction=0.032)

        _save_fig(fig_name_out)

__all__ = [
    "plot_SICB",
    "plot_SICB_ts",
    "plot_SICB_ridging_ts",
    "plot_SICB2",
]
