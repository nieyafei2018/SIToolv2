# -*- coding: utf-8 -*-
"""
Statistical analysis of missing values in sea ice reference data.

This module is the maintained home of the original
``scripts/prep/RefData_QuickCheck.py`` utilities.

Author: Yafei Nie
Created on: 2025/11/11
"""

import logging
import numpy as np
import pandas as pd
import xarray as xr
from cdo import Cdo
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scripts import utils

logger = logging.getLogger(__name__)
cdo = Cdo(options=['-f nc4 -z zip'])


def time_match(a_file, b_file, output_file):
    """Extract data from b_file based on time information from a_file.

    Args:
        a_file: Path to reference file with time information.
        b_file: Path to source file with complete data.
        output_file: Path for output file.

    Returns:
        List of valid timestamps used for matching.
    """
    with xr.open_dataset(a_file) as ds_a:
        time_var_a = next(
            (name for name in ['time', 't', 'TIME', 'Time']
             if name in ds_a.dims or name in ds_a.variables), None
        )
        if time_var_a is None:
            raise ValueError("Time variable not found in a_file")
        time_a = ds_a[time_var_a]

    try:
        time_stamps = pd.to_datetime(time_a.values)
    except Exception:
        time_stamps = pd.to_datetime(
            xr.coding.times.decode_cf_datetime(
                time_a.values,
                time_a.attrs.get('units', 'days since 1900-01-01')
            )
        )

    valid_times = [t for t in time_stamps if pd.notna(t)]
    if not valid_times:
        raise ValueError("No valid timestamps found in a_file")

    tmp = Path(output_file).parent / '_qc_temp.nc'
    try:
        cdo.monmean(input=b_file, output=str(tmp))
        with xr.open_dataset(str(tmp)) as ds:
            unique_months = {(t.year, t.month) for t in valid_times}
            time_values = pd.to_datetime(ds['time'].values)
            mask = [(t.year, t.month) in unique_months for t in time_values]
            ds.isel(time=mask).to_netcdf(output_file)
    finally:
        tmp.unlink(missing_ok=True)

    logger.info("time_match completed: %s", output_file)
    return valid_times


def stats_miss(data_file, var_name, sic_file, temp_dir=None):
    """Statistical analysis of missing values in different SIC intervals.

    Args:
        data_file: Path to data file with variable to analyze.
        var_name: Name of variable to analyze.
        sic_file: Path to sea ice concentration reference file.
        temp_dir: Temporary working directory (default: './Obs_miss_stat/').

    Returns:
        Tuple of (prop_inv, miss_prop, time_stamps).
    """
    logger.info("Counting missing measurements ...")
    temp_folder = Path(temp_dir or './Obs_miss_stat/')
    temp_folder.mkdir(parents=True, exist_ok=True)

    time_stamps = time_match(
        a_file=data_file, b_file=sic_file,
        output_file=str(temp_folder / 'SIC_temp1.nc')
    )

    utils.extract_grid(nc_file=data_file, lon_name='lon', lat_name='lat',
                       grid_file=str(temp_folder / 'grid_temp'))
    cdo.remapbil(str(temp_folder / 'grid_temp.txt'),
                 input=str(temp_folder / 'SIC_temp1.nc'),
                 output=str(temp_folder / 'SIC_temp2.nc'))

    sic_bins = np.linspace(10, 100, 19)

    with xr.open_dataset(str(temp_folder / 'SIC_temp2.nc')) as ds:
        sic = ds['siconc'].values
    with xr.open_dataset(data_file) as ds:
        data = ds[var_name].values

    sic_not_zero = (sic > 0) & (sic <= 100)
    data_is_zero_or_nan = (data == 0) | np.isnan(data)
    mask_inv = sic_not_zero & data_is_zero_or_nan

    hist_inv, _ = np.histogram(sic[mask_inv], bins=sic_bins)
    sic_hist, _ = np.histogram(sic, bins=sic_bins)
    prop_inv = hist_inv / sic_hist * 100

    miss_count = np.sum(mask_inv, axis=(1, 2))
    sic_non_zero_count = np.sum(sic_not_zero, axis=(1, 2))
    miss_prop = miss_count / sic_non_zero_count * 100

    for f in temp_folder.glob('*temp*'):
        f.unlink(missing_ok=True)

    return prop_inv, miss_prop, time_stamps


def plot_check_miss_general(data_file, var_name, hemisphere, flag,
                            sic_ref_dir, save_fig=False, temp_dir=None):
    """Plot missing value statistics for a dataset.

    Args:
        data_file: Path to data file.
        var_name: Variable name to analyze.
        hemisphere: 'nh' or 'sh'.
        flag: Identifier string for the dataset.
        sic_ref_dir: Directory containing NSIDC CDR SIC reference files.
        save_fig: Whether to save the figure.
        temp_dir: Temporary working directory.
    """
    temp_folder = Path(temp_dir or './Obs_miss_stat/')
    temp_folder.mkdir(parents=True, exist_ok=True)

    sic_ref_dir = Path(sic_ref_dir)
    if hemisphere == 'nh':
        sic_file = str(sic_ref_dir / 'NSIDC_CDR_siconc_daily_nh_19790101-20241231.nc')
    else:
        sic_file = str(sic_ref_dir / 'NSIDC_CDR_siconc_daily_sh_19790101-20241231.nc')

    prop_inv, miss_prop, dates = stats_miss(
        data_file=data_file, var_name=var_name,
        sic_file=sic_file, temp_dir=str(temp_folder)
    )

    sic_bins = np.linspace(10, 100, 19)
    xx = np.linspace(12.5, 97.5, 18)

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.tight_layout(pad=5)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    axes[0].plot(xx, prop_inv, 'r.-', label=flag)
    axes[0].set_ylabel('% of invalid values in SIC bin', fontsize=14)
    axes[0].set_xticks(sic_bins[::2])
    axes[0].set_xlim([10, 100])
    axes[0].set_xlabel('SIC interval', fontsize=14)
    axes[0].set_ylim([0, 100])
    for interval in sic_bins[1:]:
        axes[0].axvline(x=interval, color='gray', linestyle='--', alpha=0.5)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5, color='gray')
    axes[0].legend()
    axes[0].set_title(f'Missing Value Statistics - {flag}', fontsize=16)

    axes[1].plot(dates, miss_prop, 'r-', label=flag)
    axes[1].set_ylabel('% of invalid values per day', fontsize=14)
    axes[1].set_ylim([0, 100])
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    for year in range(dates[0].year, dates[-1].year + 1):
        axes[1].axvline(pd.to_datetime(f'{year}-01-01'),
                        color='gray', linestyle='--', alpha=0.5)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5, color='gray')
    axes[1].xaxis.set_minor_locator(mdates.MonthLocator())
    axes[1].xaxis.set_minor_formatter(plt.NullFormatter())

    if save_fig:
        filename = temp_folder / f'{flag}_missing_stats.png'
        plt.savefig(str(filename), dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close()
        logger.info("Saved figure: %s", filename)
    else:
        plt.show()


if __name__ == '__main__':
    pass
