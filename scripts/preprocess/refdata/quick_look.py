# -*- coding: utf-8 -*-
"""
Quick look of sea ice reference data.

This module is the maintained home of the original
``scripts/prep/RefData_QuickLook.py`` utilities.

Author: Yafei Nie
Created on: 2025/11/11
"""
import logging
import shutil
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats
from cdo import Cdo
import cartopy.crs as ccrs
from scripts import utils
from scripts import plot_figs as pf
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__name__)
cdo = Cdo(options=['-f nc4 -z zip'])

# Seasonal month indices by hemisphere
ss_dict = {
    'sh': {'Autumn': [3, 4, 5], 'Winter': [6, 7, 8], 'Spring': [9, 10, 11], 'Summer': [0, 1, 2]},
    'nh': {'Autumn': [9, 10, 11], 'Winter': [0, 1, 2], 'Spring': [3, 4, 5], 'Summer': [6, 7, 8]}
}


# ======================================================
#                        SIdrift
# ======================================================

def ql_SIdrift_clim(data_file, prod_name, period, hms, kk, output_folder='./RefData_QuickLook/'):
    """Plot seasonal climatology of sea ice drift speed and vectors.

    Args:
        data_file: Path to drift data file.
        prod_name: Product name ('NSIDC' or 'OSISAF').
        period: [start_year, end_year].
        hms: Hemisphere ('nh' or 'sh').
        kk: Quiver subsampling stride.
        output_folder: Directory for output figures.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    temp_folder = Path('./temp/')
    temp_folder.mkdir(parents=True, exist_ok=True)

    try:
        cdo.seldate(f"'{period[0]}-01-01,{period[1]}-12-31'",
                    input=data_file, output=str(temp_folder / 'temp1.nc'))

        with xr.open_dataset(str(temp_folder / 'temp1.nc')) as ds:
            lon, lat = np.array(ds['lon']), np.array(ds['lat'])

        if prod_name == 'NSIDC':
            ease = ccrs.epsg(3409 if hms == 'sh' else 3408)
            points_proj = ease.transform_points(ccrs.PlateCarree(), lon, lat)
            x0, y0 = points_proj[..., 0], points_proj[..., 1]
        else:  # OSISAF
            with xr.open_dataset(str(temp_folder / 'temp1.nc')) as ds:
                x, y = np.array(ds['xc']) * 1000, np.array(ds['yc']) * 1000
            x0, y0 = np.meshgrid(x, y)

        us = utils.cal_ss_clim_cdo(str(temp_folder / 'temp1.nc'), 'u', ss_dict[hms], threshold=None)
        vs = utils.cal_ss_clim_cdo(str(temp_folder / 'temp1.nc'), 'v', ss_dict[hms], threshold=None)

        proj = ccrs.Stereographic(central_latitude=-90 if hms == 'sh' else 90, central_longitude=0)
        fig, ax = plt.subplots(2, 2, figsize=(8, 8), subplot_kw={'projection': proj})
        ax = ax.flatten()
        seasons = list(ss_dict[hms].keys())

        for ss in range(len(seasons)):
            im = pf.polar_map(hms, ax[ss]).pcolormesh(
                x0, y0, np.sqrt(us[ss] ** 2 + vs[ss] ** 2),
                transform=proj, vmin=0, vmax=0.2, cmap=plt.get_cmap('terrain', 20))
            pf.polar_map(hms, ax[ss]).quiver(
                x0[::kk, ::kk], y0[::kk, ::kk],
                us[ss, ::kk, ::kk], vs[ss, ::kk, ::kk],
                transform=proj, color='k', width=0.004, scale=1)
            ax[ss].set_title(seasons[ss], fontsize=20)

        cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Sea Ice Speed (m/s)', fontsize=12)
        fig.tight_layout(pad=5)
        plt.subplots_adjust(wspace=0.1, hspace=0.15)

        out = output_folder / f'{prod_name}_SIdrift_{hms}_{period[0]}-{period[1]}_clim'
        plt.savefig(str(out), dpi=200, bbox_inches='tight', pad_inches=0.02)
        plt.close()
        logger.info("Saved: %s", out)
    finally:
        shutil.rmtree(temp_folder, ignore_errors=True)


def ql_SIdrift_trend(data_file, prod_name, period, hms, output_folder='./RefData_QuickLook/'):
    """Plot sea ice drift speed trend map.

    Args:
        data_file: Path to drift data file.
        prod_name: Product name ('NSIDC' or 'OSISAF').
        period: [start_year, end_year].
        hms: Hemisphere ('nh' or 'sh').
        output_folder: Directory for output figures.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    temp_folder = Path('./temp/')
    temp_folder.mkdir(parents=True, exist_ok=True)

    try:
        cdo.seldate(f"'{period[0]}-01-01,{period[1]}-12-31'",
                    input=data_file, output=str(temp_folder / 'temp1.nc'))
        cdo.monmean(input=str(temp_folder / 'temp1.nc'),
                    output=str(temp_folder / 'temp2.nc'))

        with xr.open_dataset(str(temp_folder / 'temp2.nc')) as ds:
            u_m, v_m = np.array(ds['u']), np.array(ds['v'])
            lon, lat = np.array(ds['lon']), np.array(ds['lat'])

        nt, nx, ny = u_m.shape

        if prod_name == 'NSIDC':
            ease = ccrs.epsg(3409 if hms == 'sh' else 3408)
            points_proj = ease.transform_points(ccrs.PlateCarree(), lon, lat)
            x0, y0 = points_proj[..., 0], points_proj[..., 1]
        else:
            with xr.open_dataset(str(temp_folder / 'temp2.nc')) as ds:
                x, y = np.array(ds['xc']) * 1000, np.array(ds['yc']) * 1000
            x0, y0 = np.meshgrid(x, y)

        speed_m = np.sqrt(u_m ** 2 + v_m ** 2)
        speed_clim = np.array([np.nanmean(speed_m[m::12], axis=0) for m in range(12)])
        speed_ano = np.array([speed_m[j] - speed_clim[j % 12] for j in range(nt)])

        months = np.arange(1, nt + 1)
        if nt / 12 < 15:
            logger.warning("Years analyzed < 15; trend estimates have considerable uncertainty.")

        speed_ano_tr = np.full((nx, ny), np.nan)
        speed_ano_tr_p = np.full((nx, ny), np.nan)
        for jx in range(nx):
            for jy in range(ny):
                ano = speed_ano[:, jx, jy]
                if np.sum(np.isnan(ano)) > 0.5 * nt:
                    # More than half NaN values, skip
                    speed_ano_tr[jx, jy] = np.nan
                    speed_ano_tr_p[jx, jy] = 1.0
                else:
                    # Remove NaN data points
                    valid = ~np.isnan(ano)
                    slope, _, _, p_value, _ = stats.linregress(months[valid], ano[valid])
                    speed_ano_tr[jx, jy] = slope * 12 * 10
                    speed_ano_tr_p[jx, jy] = p_value

        proj = ccrs.Stereographic(central_latitude=-90 if hms == 'sh' else 90, central_longitude=0)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': proj})
        im = pf.polar_map(hms, ax).pcolormesh(
            x0, y0, speed_ano_tr, transform=proj,
            vmin=-0.01, vmax=0.01, cmap=plt.get_cmap('coolwarm', 20))

        x1, y1 = x0.copy(), y0.copy()
        x1[speed_ano_tr_p > 0.05] = np.nan
        pf.polar_map(hms, ax).plot(x1, y1, color='black', linewidth=0.3)

        cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Sea Ice Speed Trend (m/s / decade)', fontsize=12)
        fig.tight_layout(pad=5)
        plt.subplots_adjust(wspace=0.1, hspace=0.15)

        out = output_folder / f'{prod_name}_SIdrift_{hms}_{period[0]}-{period[1]}_trend'
        plt.savefig(str(out), dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close()
        logger.info("Saved: %s", out)
    finally:
        shutil.rmtree(temp_folder, ignore_errors=True)


# ======================================================
#                        SNdepth
# ======================================================

def ql_SNdepth_clim_sh(sndepth_ref_dir, output_folder='./RefData_QuickLook/'):
    """Plot SH snow depth seasonal climatology (CTOH and MPMR).

    Args:
        sndepth_ref_dir: Directory containing SNdepth reference files.
        output_folder: Directory for output figures.
    """
    sndepth_ref_dir = Path(sndepth_ref_dir)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    MPMR_file = str(sndepth_ref_dir / 'MPMR_SNdepth_sh_20020601-20200531_monmean.nc')
    CTOH_file = str(sndepth_ref_dir / 'CTOH_sndepth_monthly_sh_199404-202306.nc')

    proj = ccrs.Stereographic(central_latitude=-90, central_longitude=0)
    fig, ax = plt.subplots(2, 4, figsize=(15, 7), subplot_kw={'projection': proj})

    with xr.open_dataset(CTOH_file) as ds:
        lon_CTOH, lat_CTOH = ds['lon'].values, ds['lat'].values
        sd_CTOH = ds['snow_depth_eff']
    with xr.open_dataset(MPMR_file) as ds:
        lon_MPMR, lat_MPMR = ds['lon'].values, ds['lat'].values
        sd_MPMR = ds['snow_depth_eff']

    # Calculate seasonal climatology for both datasets
    sd_CTOH_seasonal = utils.cal_ss_clim(sd_CTOH, ss_dict['sh'])
    sd_MPMR_seasonal = utils.cal_ss_clim(sd_MPMR, ss_dict['sh'])

    seasons = list(ss_dict['sh'].keys())
    for ii, season in enumerate(seasons):
        for row, (lon, lat, data) in enumerate([
            (lon_CTOH, lat_CTOH, sd_CTOH_seasonal[season]),
            (lon_MPMR, lat_MPMR, sd_MPMR_seasonal[season]),
        ]):
            mesh = pf.polar_map('sh', ax[row, ii]).pcolormesh(
                lon, lat, data, vmin=0, vmax=0.6,
                cmap=plt.get_cmap('RdYlBu_r', 10), transform=ccrs.PlateCarree())
            divider = make_axes_locatable(ax[row, ii])
            cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
            plt.colorbar(mesh, cax=cax)
            ax[row, ii].set_title(season)

    fig.tight_layout(pad=5)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig(str(output_folder / 'sd_pattern_sh.png'), dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def ql_SNdepth_sh(sndepth_ref_dir, output_folder='./RefData_QuickLook/'):
    """Plot SH snow depth time series and seasonal cycle (CTOH and MPMR).

    Args:
        sndepth_ref_dir: Directory containing SNdepth reference files.
        output_folder: Directory for output figures.
    """
    sndepth_ref_dir = Path(sndepth_ref_dir)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    MPMR_file = str(sndepth_ref_dir / 'MPMR_SNdepth_sh_20020601-20200531_monmean.nc')
    CTOH_file = str(sndepth_ref_dir / 'CTOH_sndepth_monthly_sh_199404-202306.nc')

    with xr.open_dataset(MPMR_file) as ds:
        sd_MPMR = ds['snow_depth_eff'].where(ds['snow_depth_eff'] > 0, np.nan)
        time_MPMR = ds['time']
    with xr.open_dataset(CTOH_file) as ds:
        sd_CTOH = ds['snow_depth_eff'].where(ds['snow_depth_eff'] > 0, np.nan)
        time_CTOH = ds['time']

    sd_MPMR_ts = sd_MPMR.mean(dim=['x', 'y'], skipna=True)
    sd_MPMR_clim = sd_MPMR_ts.groupby('time.month').mean(skipna=True)
    sd_MPMR_anom = sd_MPMR_ts.groupby('time.month') - sd_MPMR_clim
    sd_CTOH_ts = sd_CTOH.mean(dim=['x', 'y'], skipna=True)
    sd_CTOH_clim = sd_CTOH_ts.groupby('time.month').mean(skipna=True)
    sd_CTOH_anom = sd_CTOH_ts.groupby('time.month') - sd_CTOH_clim

    fig, ax = plt.subplots(3, 1, figsize=(12, 10))
    ax[0].plot(time_MPMR.values, sd_MPMR_ts, 'k-', label='MPMR')
    ax[0].plot(time_CTOH.values, sd_CTOH_ts, 'b-', label='CTOH')
    ax[0].set_ylabel('Mean value (m)')
    ax[0].legend()
    ax[1].plot(time_MPMR.values, sd_MPMR_anom, 'k-')
    ax[1].plot(time_CTOH.values, sd_CTOH_anom, 'b-')
    ax[1].set_ylabel('Anomaly (m)')
    for ii in range(2):
        ax[ii].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax[ii].xaxis.set_major_locator(mdates.AutoDateLocator())
        ax[ii].grid(True, alpha=0.5)
    ax[2].plot(np.arange(12), sd_MPMR_clim, 'k-')
    ax[2].plot(np.arange(12), sd_CTOH_clim, 'b-')
    ax[2].set_ylabel('Seasonal cycle (m)')
    plt.savefig(str(output_folder / 'sd_sh.png'), dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def ql_SNdepth_clim_nh(sndepth_ref_dir, output_folder='./RefData_QuickLook/'):
    """Plot NH snow depth seasonal climatology (CTOH and SnowModel-LG).

    Args:
        sndepth_ref_dir: Directory containing SNdepth reference files.
        output_folder: Directory for output figures.
    """
    sndepth_ref_dir = Path(sndepth_ref_dir)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    temp_folder = Path('./temp/')
    temp_folder.mkdir(parents=True, exist_ok=True)

    try:
        CTOH_file = str(sndepth_ref_dir / 'CTOH_sndepth_monthly_nh_199410-202304.nc')
        LG_file_E5 = str(sndepth_ref_dir / 'SnowModel-LG_nh_snod_ERA5_v01_19800801-20210731.nc')
        LG_file_M2 = str(sndepth_ref_dir / 'SnowModel-LG_nh_snod_MERRA2_v01_19800801-20210731.nc')

        cdo.monmean(input=LG_file_E5, output=str(temp_folder / 'LG_ERA5_monmean.nc'))
        cdo.monmean(input=LG_file_M2, output=str(temp_folder / 'LG_MERRA2_monmean.nc'))

        ss_num = {'Autumn': [9, 10, 11], 'Winter': [0, 1, 2, 3], 'Spring': [4, 5, 6], 'Summer': [7, 8]}
        seasons = list(ss_num.keys())

        proj = ccrs.Stereographic(central_latitude=90, central_longitude=0)
        fig, ax = plt.subplots(3, 4, figsize=(15, 10.5), subplot_kw={'projection': proj})

        with xr.open_dataset(CTOH_file) as ds:
            lon_CTOH, lat_CTOH = ds['lon'].values, ds['lat'].values
            sd_CTOH = ds['snow_depth_eff']
        with xr.open_dataset(str(temp_folder / 'LG_ERA5_monmean.nc')) as ds:
            lon_LG, lat_LG = ds['lon'].values, ds['lat'].values
            sd_LG_E5 = ds['snow_depth_eff']
        with xr.open_dataset(str(temp_folder / 'LG_MERRA2_monmean.nc')) as ds:
            sd_LG_M2 = ds['snow_depth_eff']

        # Calculate seasonal climatology for all datasets
        sd_CTOH_seasonal = utils.cal_ss_clim(sd_CTOH, ss_num, 0)
        sd_LG_E5_seasonal = utils.cal_ss_clim(sd_LG_E5, ss_num, 0)
        sd_LG_M2_seasonal = utils.cal_ss_clim(sd_LG_M2, ss_num, 0)

        for ii, season in enumerate(seasons):
            for row, (lon, lat, data) in enumerate([
                (lon_CTOH, lat_CTOH, sd_CTOH_seasonal[season]),
                (lon_LG, lat_LG, sd_LG_E5_seasonal[season].where(sd_LG_E5_seasonal[season] > 0, np.nan)),
                (lon_LG, lat_LG, sd_LG_M2_seasonal[season].where(sd_LG_M2_seasonal[season] > 0, np.nan)),
            ]):
                mesh = pf.polar_map('sh', ax[row, ii]).pcolormesh(
                    lon, lat, data, vmin=0, vmax=0.4,
                    cmap=plt.get_cmap('RdYlBu_r', 10), transform=ccrs.PlateCarree())
                divider = make_axes_locatable(ax[row, ii])
                cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
                plt.colorbar(mesh, cax=cax)
                ax[row, ii].set_title(season)

        fig.tight_layout(pad=5)
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        plt.savefig(str(output_folder / 'sd_pattern_nh.png'), dpi=200, bbox_inches='tight', pad_inches=0.02)
        plt.close()
    finally:
        shutil.rmtree(temp_folder, ignore_errors=True)


def ql_SNdepth_nh(sndepth_ref_dir, output_folder='./RefData_QuickLook/'):
    """Plot NH snow depth time series and seasonal cycle.

    Args:
        sndepth_ref_dir: Directory containing SNdepth reference files.
        output_folder: Directory for output figures.
    """
    sndepth_ref_dir = Path(sndepth_ref_dir)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    CTOH_file = str(sndepth_ref_dir / 'CTOH_sndepth_monthly_nh_199410-202304.nc')
    LG_file_E5 = str(sndepth_ref_dir / 'SnowModel-LG_nh_snod_ERA5_v01_19800801-20210731.nc')
    LG_file_M2 = str(sndepth_ref_dir / 'SnowModel-LG_nh_snod_MERRA2_v01_19800801-20210731.nc')

    full_time = pd.date_range('1994-10-01', '2023-04-01', freq='MS')
    with xr.open_dataset(CTOH_file) as ds:
        sd_CTOH = ds['snow_depth_eff'].where(ds['snow_depth_eff'] > 0, np.nan)
        time_CTOH = ds['time']

    sd_CTOH_ts_full = np.full(len(full_time), np.nan, dtype=np.float32)
    for i, t in enumerate(full_time):
        mask = (time_CTOH.dt.year.values == t.year) & (time_CTOH.dt.month.values == t.month)
        if np.any(mask):
            sd_CTOH_ts_full[i] = np.nanmean(sd_CTOH.isel(time=np.where(mask)[0][0]).values)

    sd_CTOH_ts = xr.DataArray(sd_CTOH_ts_full, coords={'time': full_time}, dims=['time'])
    sd_CTOH_clim = sd_CTOH_ts.groupby('time.month').mean(skipna=True)
    sd_CTOH_anom = sd_CTOH_ts.groupby('time.month') - sd_CTOH_clim

    full_time_LG = pd.date_range('1980-08-01', '2021-07-31', freq='MS')
    with xr.open_dataset(LG_file_E5) as ds:
        sd_LG_E5 = ds['snow_depth_eff']
        time_LG = ds['time']
    with xr.open_dataset(LG_file_M2) as ds:
        sd_LG_M2 = ds['snow_depth_eff']

    sd_LG_E5_ts_full = np.full(len(full_time_LG), np.nan, dtype=np.float32)
    sd_LG_M2_ts_full = np.full(len(full_time_LG), np.nan, dtype=np.float32)
    for i, t in enumerate(full_time_LG):
        mask = (time_LG.dt.year.values == t.year) & (time_LG.dt.month.values == t.month)
        if np.any(mask):
            idx = np.where(mask)[0][0]
            sd_LG_E5_ts_full[i] = np.nanmean(sd_LG_E5.isel(time=idx).values)
            sd_LG_M2_ts_full[i] = np.nanmean(sd_LG_M2.isel(time=idx).values)

    sd_LG_E5_ts = xr.DataArray(sd_LG_E5_ts_full, coords={'time': full_time_LG}, dims=['time'])
    sd_LG_M2_ts = xr.DataArray(sd_LG_M2_ts_full, coords={'time': full_time_LG}, dims=['time'])
    sd_LG_E5_clim = sd_LG_E5_ts.groupby('time.month').mean(skipna=True)
    sd_LG_E5_anom = sd_LG_E5_ts.groupby('time.month') - sd_LG_E5_clim
    sd_LG_M2_clim = sd_LG_M2_ts.groupby('time.month').mean(skipna=True)
    sd_LG_M2_anom = sd_LG_M2_ts.groupby('time.month') - sd_LG_M2_clim

    fig, ax = plt.subplots(3, 1, figsize=(12, 10))
    ax[0].plot(full_time, sd_CTOH_ts, 'b-')
    ax[0].plot(full_time_LG, sd_LG_E5_ts, 'r-')
    ax[0].plot(full_time_LG, sd_LG_M2_ts, 'm-')
    ax[0].set_ylabel('Mean value (m)')
    ax[1].plot(full_time, sd_CTOH_anom, 'b-')
    ax[1].plot(full_time_LG, sd_LG_E5_anom, 'r-')
    ax[1].plot(full_time_LG, sd_LG_M2_anom, 'm-')
    ax[1].set_ylabel('Anomaly (m)')
    for ii in range(2):
        ax[ii].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax[ii].xaxis.set_major_locator(mdates.AutoDateLocator())
        ax[ii].grid(True, alpha=0.5)
    ax[2].plot(np.arange(12), sd_CTOH_clim, 'b-')
    ax[2].plot(np.arange(12), sd_LG_E5_clim, 'r-')
    ax[2].plot(np.arange(12), sd_LG_M2_clim, 'm-')
    ax[2].set_ylabel('Seasonal cycle (m)')
    plt.savefig(str(output_folder / 'sd_nh.png'), dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()


# ======================================================
#                        SIthick
# ======================================================

def ql_SIthick_clim_sh(sithick_ref_dir, output_folder='./RefData_QuickLook/'):
    """Plot SH sea ice thickness seasonal climatology (CTOH and GIOMAS).

    Args:
        sithick_ref_dir: Directory containing SIthick reference files.
        output_folder: Directory for output figures.
    """
    sithick_ref_dir = Path(sithick_ref_dir)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    CTOH_file = str(sithick_ref_dir / 'CTOH_sithick_monthly_sh_199404-202306.nc')
    GIOMAS_file = str(sithick_ref_dir / 'GIOMAS_sh_sithick_mon_197901-202312.nc')

    proj = ccrs.Stereographic(central_latitude=-90, central_longitude=0)
    fig, ax = plt.subplots(2, 4, figsize=(15, 7), subplot_kw={'projection': proj})

    with xr.open_dataset(CTOH_file) as ds:
        lon_CTOH, lat_CTOH = ds['lon'].values, ds['lat'].values
        sit_CTOH = ds['sithick_eff']
    with xr.open_dataset(GIOMAS_file) as ds:
        lon_GIOMAS, lat_GIOMAS = ds['lon'].values, ds['lat'].values
        sit_GIOMAS = ds['sithick_eff']

    # Calculate seasonal climatology for both datasets
    sit_CTOH_seasonal = utils.cal_ss_clim(sit_CTOH, ss_dict['sh'], 0.001)
    sit_GIOMAS_seasonal = utils.cal_ss_clim(sit_GIOMAS, ss_dict['sh'], 0.001)

    seasons = list(ss_dict['sh'].keys())
    for ii, season in enumerate(seasons):
        mesh = pf.polar_map('sh', ax[0, ii]).pcolormesh(
            lon_CTOH, lat_CTOH, sit_CTOH_seasonal[season],
            vmin=0, vmax=3, cmap=plt.get_cmap('RdYlBu_r', 10), transform=ccrs.PlateCarree())
        divider = make_axes_locatable(ax[0, ii])
        cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
        plt.colorbar(mesh, cax=cax)
        ax[0, ii].set_title(season)

        data = sit_GIOMAS_seasonal[season].where(sit_GIOMAS_seasonal[season] > 0.05, np.nan)
        mesh = pf.polar_map('sh', ax[1, ii]).pcolormesh(
            lon_GIOMAS, lat_GIOMAS, data,
            vmin=0, vmax=3, cmap=plt.get_cmap('RdYlBu_r', 10), transform=ccrs.PlateCarree())
        divider = make_axes_locatable(ax[1, ii])
        cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
        plt.colorbar(mesh, cax=cax)
        ax[1, ii].set_title(season)

    fig.tight_layout(pad=5)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig(str(output_folder / 'sit_pattern_sh.png'), dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def ql_SIthick_sh(sithick_ref_dir, output_folder='./RefData_QuickLook/'):
    """Plot SH sea ice thickness time series and seasonal cycle.

    Args:
        sithick_ref_dir: Directory containing SIthick reference files.
        output_folder: Directory for output figures.
    """
    sithick_ref_dir = Path(sithick_ref_dir)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    CTOH_file = str(sithick_ref_dir / 'CTOH_sithick_monthly_sh_199404-202306.nc')
    GIOMAS_file = str(sithick_ref_dir / 'GIOMAS_sh_sithick_mon_197901-202312.nc')

    with xr.open_dataset(GIOMAS_file) as ds:
        sit_GIOMAS = ds['sithick_eff'].where(ds['sithick_eff'] > 0.05, np.nan)
        time_GIOMAS = ds['time']
    with xr.open_dataset(CTOH_file) as ds:
        sit_CTOH = ds['sithick_eff'].where(ds['sithick_eff'] > 0, np.nan)
        time_CTOH = ds['time']

    sit_GIOMAS_ts = sit_GIOMAS.mean(dim=['x', 'y'], skipna=True)
    sit_GIOMAS_clim = sit_GIOMAS_ts.groupby('time.month').mean(skipna=True)
    sit_GIOMAS_anom = sit_GIOMAS_ts.groupby('time.month') - sit_GIOMAS_clim
    sit_CTOH_ts = sit_CTOH.mean(dim=['x', 'y'], skipna=True)
    sit_CTOH_clim = sit_CTOH_ts.groupby('time.month').mean(skipna=True)
    sit_CTOH_anom = sit_CTOH_ts.groupby('time.month') - sit_CTOH_clim

    fig, ax = plt.subplots(3, 1, figsize=(12, 10))
    ax[0].plot(time_GIOMAS.values, sit_GIOMAS_ts, 'k-', label='GIOMAS')
    ax[0].plot(time_CTOH.values, sit_CTOH_ts, 'b-', label='CTOH')
    ax[0].set_ylabel('Mean value (m)')
    ax[0].legend()
    ax[1].plot(time_GIOMAS.values, sit_GIOMAS_anom, 'k-')
    ax[1].plot(time_CTOH.values, sit_CTOH_anom, 'b-')
    ax[1].set_ylabel('Anomaly (m)')
    for ii in range(2):
        ax[ii].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax[ii].xaxis.set_major_locator(mdates.AutoDateLocator())
        ax[ii].grid(True, alpha=0.5)
    ax[2].plot(np.arange(12), sit_GIOMAS_clim, 'k-')
    ax[2].plot(np.arange(12), sit_CTOH_clim, 'b-')
    ax[2].set_ylabel('Seasonal cycle (m)')
    plt.savefig(str(output_folder / 'sit_sh.png'), dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def ql_SIthick_clim_nh(sithick_ref_dir, output_folder='./RefData_QuickLook/'):
    """Plot NH sea ice thickness seasonal climatology (CTOH, PIOMAS, CSBD, TOPAZ).

    Args:
        sithick_ref_dir: Directory containing SIthick reference files.
        output_folder: Directory for output figures.
    """
    sithick_ref_dir = Path(sithick_ref_dir)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    ss_num = {'Autumn': [9, 10, 11], 'Winter': [0, 1, 2, 3], 'Spring': [4, 5, 6], 'Summer': [7, 8]}
    seasons = list(ss_num.keys())

    files = {
        'CTOH':   sithick_ref_dir / 'CTOH_sithick_monthly_nh_199410-202304.nc',
        'PIOMAS': sithick_ref_dir / 'PIOMAS_nh_sithick_mon_197901-202312.nc',
        'CSBD':   sithick_ref_dir / 'CSBD_ubristol_cs2_sit_nh_80km_v1p7_monthly_201011-202007.nc',
        'TOPAZ':  sithick_ref_dir / 'TOPAZ4b_nh_sithick_mon_199101-202312.nc',
    }
    datasets = {}
    for name, fpath in files.items():
        with xr.open_dataset(str(fpath)) as ds:
            datasets[name] = {
                'lon': ds['lon'].values, 'lat': ds['lat'].values,
                'seasonal': utils.cal_ss_clim(ds['sithick_eff'], ss_num, 0.)
            }

    proj = ccrs.Stereographic(central_latitude=90, central_longitude=0)
    fig, ax = plt.subplots(4, 4, figsize=(16, 15), subplot_kw={'projection': proj})

    for ii, season in enumerate(seasons):
        for row, name in enumerate(['CTOH', 'PIOMAS', 'CSBD', 'TOPAZ']):
            d = datasets[name]
            mesh = pf.polar_map('nh', ax[row, ii]).pcolormesh(
                d['lon'], d['lat'], d['seasonal'][season],
                vmin=0, vmax=3, cmap=plt.get_cmap('RdYlBu_r', 10), transform=ccrs.PlateCarree())
            divider = make_axes_locatable(ax[row, ii])
            cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
            plt.colorbar(mesh, cax=cax)
            ax[row, ii].set_title(season)

    fig.tight_layout(pad=5)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig(str(output_folder / 'sit_pattern_nh.png'), dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()


if __name__ == '__main__':
    pass
