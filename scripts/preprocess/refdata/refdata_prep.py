# -*- coding: utf-8 -*-
"""
Data preprocessing for sea ice reference data.

This module is the maintained home of the original
``scripts/prep/RefData_Prep.py`` implementation.
It was moved under ``scripts/preprocess/refdata`` so all preprocessing
logic (online and offline) lives under one package namespace.

Standard variable naming convention:
------------------------------------
    SIconc: <siconc>; <siconc_unc>
    SIdrift: <u>, <v>; <u_unc>, <v_unc>
    SNdepth: <snow_depth>; <snow_depth_unc>
        For ctoh_v2: <STAT_snow_depth_q025>, <STAT_snow_depth_q975>
    SIthick: <sithick>; <sithick_eff>; <sithick_unc>
        For ctoh_v2: <STAT_sea_ice_thick_q025>, <STAT_sea_ice_thick_q975>

Created on 2023/7/12
"""
import logging
import struct
import time
import shutil
import pyproj
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from cdo import Cdo
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from scripts import utils

logger = logging.getLogger(__name__)
cdo = Cdo(options=['-f nc4 -z zip'])

_CDO_OPTS = ['-f nc4 -z zip']


# ---------------------------------------------------------------------------
# Module-level helpers for intra-task ThreadPoolExecutor workers.
# Each creates its own Cdo() instance to avoid shared-object contention.
# ---------------------------------------------------------------------------

def _nsidc_sic_selvar(file_str: str, temp_folder_str: str) -> None:
    """Extract SIC variables from one NSIDC CDR annual file."""
    file = Path(file_str)
    out = Path(temp_folder_str) / file.name
    if not out.exists():
        Cdo(options=_CDO_OPTS).selvar(
            "'cdr_seaice_conc,cdr_seaice_conc_stdev'",
            input=str(file),
            output=str(out),
        )


def _osi_sic_year(data_path_str: str, hemisphere: str,
                  temp_folder_str: str, year: int) -> None:
    """Process one year of OSI-450 SIC data: mergetime then selname."""
    data_path = Path(data_path_str)
    temp_folder = Path(temp_folder_str)
    file_list = sorted(data_path.glob(f'{year}/*/*{hemisphere}*'))
    out_year = temp_folder / f'temp1_{year}.nc'
    if not file_list or out_year.exists():
        return
    _cdo = Cdo(options=_CDO_OPTS)
    _cdo.mergetime(input=[str(f) for f in file_list], output=str(out_year))
    _cdo.selname(
        "'ice_conc,total_standard_uncertainty'",
        input=str(out_year),
        output=str(temp_folder / f'temp1_{year}_ice.nc'),
    )


def _mpmr_year(year_dir_str: str, unc_root_str: str, temp_folder_str: str,
               longitude: np.ndarray, latitude: np.ndarray,
               grid_txt_str: str) -> str | None:
    """Process one year of MPMR TIFFs → gridded NetCDF.

    Returns the path to the output *_pg.nc file, or None if no TIFFs found.
    """
    year_dir = Path(year_dir_str)
    unc_root = Path(unc_root_str)
    temp_folder = Path(temp_folder_str)
    year = year_dir.name

    year_out_nc = temp_folder / f'temp_{year}_pg.nc'
    if year_out_nc.exists():
        return str(year_out_nc)

    sd_files = sorted(year_dir.glob('*_SD.tif'))
    if not sd_files:
        return None

    dates = [datetime.strptime(f.stem.split('_SD')[0], '%Y%m%d') for f in sd_files]
    unc_files = [
        unc_root / year / (f.stem.replace('_SD', '_SDunc') + '.tif')
        for f in sd_files
    ]

    sd_data = []
    for f in sd_files:
        with rasterio.open(str(f)) as src:
            sd_data.append(src.read(1))
    sd_data = np.array(sd_data, dtype=np.float32) / 100

    unc_data = []
    for f, ref in zip(unc_files, sd_data):
        if f.exists():
            with rasterio.open(str(f)) as src:
                unc_data.append(src.read(1))
        else:
            unc_data.append(np.full(ref.shape, np.nan, dtype=np.float32))
    unc_data = np.array(unc_data, dtype=np.float32) / 100

    ds_year = xr.Dataset({
        'snow_depth':     (['time', 'y', 'x'], sd_data),
        'snow_depth_unc': (['time', 'y', 'x'], unc_data),
        'longitude':      (['y', 'x'], longitude),
        'latitude':       (['y', 'x'], latitude),
    }, coords={'time': dates})
    encoding = {v: {'zlib': True, 'complevel': 4} for v in ds_year.data_vars}
    temp_raw_nc = temp_folder / f'temp_{year}_raw.nc'
    ds_year.to_netcdf(str(temp_raw_nc), encoding=encoding)
    del ds_year, sd_data, unc_data

    _cdo = Cdo(options=_CDO_OPTS)
    temp_sg_nc = temp_folder / f'temp_{year}_sg.nc'
    _cdo.setgrid(grid_txt_str, input=str(temp_raw_nc), output=str(temp_sg_nc))
    temp_raw_nc.unlink(missing_ok=True)
    _cdo.delname("'longitude,latitude'", input=str(temp_sg_nc), output=str(year_out_nc))
    temp_sg_nc.unlink(missing_ok=True)

    return str(year_out_nc)


class ReferenceDataManager:
    """Manages preprocessing of sea ice reference datasets.

    Args:
        ref_data_dir: Root directory for processed reference data output.
    """

    # -----------------------------------------------------------------------------
    # Function: `__init__`.
    # Purpose: Handle init workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    def __init__(self, ref_data_dir, inner_jobs: int = 1):
        self.ref_data_dir = Path(ref_data_dir)
        self.sic_dir = self.ref_data_dir / 'SIconc'
        self.drift_dir = self.ref_data_dir / 'SIdrift'
        self.sndepth_dir = self.ref_data_dir / 'SNdepth'
        self.thick_dir = self.ref_data_dir / 'SIthick'
        self.inner_jobs = max(1, int(inner_jobs))
        for d in [self.ref_data_dir, self.sic_dir, self.drift_dir,
                  self.sndepth_dir, self.thick_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ======================================================
    #                        SIconc
    # ======================================================

    def prep_NSIDC_SIconc(self, data_path, hemisphere):
        """Process and merge NSIDC CDR daily sea ice concentration files.

        Args:
            data_path: Directory of /G02202_V5 folder.
            hemisphere: Target hemisphere: 'nh' or 'sh'.

        Data Info
        ---------
        NSIDC CDR v5 (G02202)
        Avaliable at: https://nsidc.org/data/g02202/versions/5
        Can be downloaded from NOAA@NSIDC HTTPS File System:
            https://noaadata.apps.nsidc.org/NOAA/G02202_V5/south/aggregate
            https://noaadata.apps.nsidc.org/NOAA/G02202_V5/north/aggregate

        Directory tree
        --------------
        |--G02202_V5
            |--north
                |--aggregate
                    |--seaice_conc_daily_nh_1979_v04r00.nc
                    ...
                    |--seaice_conc_daily_nh_2023_v04r00.nc
            |-- south
                |--aggregate
                    |--seaice_conc_daily_sh_1979_v04r00.nc
                    ...
                    |--seaice_conc_daily_sh_2023_v04r00.nc
        """
        logger.info("Preprocessing NSIDC %s daily SIC ...", hemisphere)
        out_file = self.sic_dir / f'NSIDC_CDR_siconc_daily_{hemisphere}_19790101-20241231.nc'
        if out_file.exists():
            logger.info("  -> output already exists, skipping.")
            return
        data_path = Path(data_path)
        temp_folder = self.sic_dir / f'temp_NSIDC_{hemisphere}'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            hemisphere_dir = 'north' if hemisphere == 'nh' else 'south'
            # --- Standardization: extract two SIC variables per annual file ---
            nc_files = list(data_path.glob(f'{hemisphere_dir}/aggregate/*.nc'))
            if self.inner_jobs > 1:
                logger.debug("NSIDC SIconc %s: processing %d files with %d threads.",
                             hemisphere, len(nc_files), self.inner_jobs)
                with ThreadPoolExecutor(max_workers=self.inner_jobs) as pool:
                    futs = [pool.submit(_nsidc_sic_selvar, str(f), str(temp_folder))
                            for f in nc_files]
                    for fut in as_completed(futs):
                        fut.result()
            else:
                for file in nc_files:
                    _nsidc_sic_selvar(str(file), str(temp_folder))

            # --- Merge to one file ---
            cdo.mergetime(input=str(temp_folder / '*.nc'),
                          output=str(temp_folder / 'temp1.nc'))
            # --- set range to 0-100 ---
            cdo.expr("'siconc=cdr_seaice_conc*100;siconc_std=cdr_seaice_conc_stdev*100'",
                     input=str(temp_folder / 'temp1.nc'),
                     output=str(temp_folder / 'temp2.nc'))

            # --- Set grid ---
            grid_file = str(self.ref_data_dir / 'Auxiliary' /
                            'NSIDC0771_LatLon_PS_S25km_v1.0.nc')
            utils.extract_grid(nc_file=grid_file,
                               lon_name='longitude', lat_name='latitude',
                               grid_file=str(temp_folder / f'NSIDC_CDR_{hemisphere}'))
            cdo.setgrid(str(temp_folder / f'NSIDC_CDR_{hemisphere}.txt'),
                        input=str(temp_folder / 'temp2.nc'),
                        output=str(self.sic_dir /
                                   f'NSIDC_CDR_siconc_daily_{hemisphere}_19790101-20241231.nc'))
        finally:
            shutil.rmtree(temp_folder, ignore_errors=True)

        logger.info("NSIDC %s daily SIC completed in %.0fs.", hemisphere, time.time() - ts)

    # -----------------------------------------------------------------------------
    # Function: `prep_OSI_SIconc`.
    # Purpose: Handle prep OSI SIconc workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    def prep_OSI_SIconc(self, data_path, hemisphere):
        """Process and merge OSI-450 daily sea ice concentration files.

        Args:
            data_path: Directory of /v3p1 folder.
            hemisphere: Target hemisphere: 'nh' or 'sh'.

        Data Info
        ---------
        https://doi.org/10.15770/EUM_SAF_OSI_0013
        https://doi.org/10.15770/EUM_SAF_OSI_0014

        Downloaded from OSI SAF FTP server:
            anonymous@osisaf.met.no
        Directory:
            /reprocessed/ice/conc/v3p1            (1978-2020)
            /reprocessed/conc-cont-reproc/v3p1    (2021-onwards)

        Directory tree
        --------------
        |--OSISAF
            |--v3p1
                |--1979
                    |--01
                        |--ice_conc_nh_ease2-250_cdr-v3p0_197901021200.nc
                        |--ice_conc_nh_ease2-250_cdr-v3p0_197901041200.nc
                        ...
                    ...
                    |--12
                ...
                |--2024
        """
        logger.info("Preprocessing OSI %s daily SIC ...", hemisphere)
        out_file = self.sic_dir / f'OSI-450_siconc_daily_{hemisphere}_19790101-20231231.nc'
        if out_file.exists():
            logger.info("  -> output already exists, skipping.")
            return
        data_path = Path(data_path)
        temp_folder = self.sic_dir / f'temp_OSI_{hemisphere}'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            # --- Per-year: mergetime + selname (years are independent) ---
            years = list(range(1979, 2025))
            if self.inner_jobs > 1:
                logger.debug("OSI SIconc %s: processing %d years with %d threads.",
                             hemisphere, len(years), self.inner_jobs)
                with ThreadPoolExecutor(max_workers=self.inner_jobs) as pool:
                    futs = [pool.submit(_osi_sic_year, str(data_path), hemisphere,
                                        str(temp_folder), y) for y in years]
                    for fut in as_completed(futs):
                        fut.result()
            else:
                for year in years:
                    logger.debug("Processing year: %d", year)
                    _osi_sic_year(str(data_path), hemisphere, str(temp_folder), year)

            cdo.mergetime(input=str(temp_folder / 'temp1_*_ice.nc'),
                          output=str(temp_folder / 'temp1.nc'))

            # --- Set grid ---
            sample_file = next(data_path.glob(f'*/*/*{hemisphere}*'))
            utils.extract_grid(nc_file=str(sample_file),
                               lon_name='lon', lat_name='lat',
                               grid_file=str(temp_folder / f'OSI_CDR_{hemisphere}'))
            cdo.setgrid(str(temp_folder / f'OSI_CDR_{hemisphere}.txt'),
                        input=str(temp_folder / 'temp1.nc'),
                        output=str(temp_folder / 'temp2.nc'))
            # --- Standardization: change variables' name ---
            cdo.chname("'ice_conc,siconc,total_standard_uncertainty,siconc_std'",
                       input=str(temp_folder / 'temp2.nc'),
                       output=str(temp_folder / 'temp3.nc'))
            # --- Remove the duplicated data ---
            cdo.daymean(input=str(temp_folder / 'temp3.nc'),
                        output=str(self.sic_dir /
                                   f'OSI-450_siconc_daily_{hemisphere}_19790101-20231231.nc'))
        finally:
            shutil.rmtree(temp_folder, ignore_errors=True)

        logger.info("OSI %s daily SIC completed in %.0fs.", hemisphere, time.time() - ts)

    # ======================================================
    #                        SIdrift
    # ======================================================

    def prep_NSIDC_SIdrift(self, data_path, hemisphere):
        """Process and merge NSIDC Polar Pathfinder daily sea ice drift files.

        Args:
            data_path: Directory of /Polar_Pathfinder_sea_ice_motion folder.
            hemisphere: Target hemisphere: 'nh' or 'sh'.

        Data Info
        ---------
        Avaliable at: https://doi.org/10.5067/INAWUWO7QH7B
        Can be downloaded from NOAA@NSIDC HTTPS File System:
            https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0116_icemotion_vectors_v4/

        Spatial Reference System(s):
            NSIDC EASE-Grid North (EPSG:3408)
            NSIDC EASE-Grid South (EPSG:3409)

        Directory tree
        --------------
        |--Polar_Pathfinder_sea_ice_motion
            |--north
                |--daily
                    |--icemotion_daily_nh_25km_19790101_19791231_v4.1.nc
                    |--icemotion_daily_nh_25km_19800101_19801231_v4.1.nc
                    ...
            |--south
                |--daily
                    |--icemotion_daily_sh_25km_19790101_19791231_v4.1.nc
                    |--icemotion_daily_sh_25km_19800101_19801231_v4.1.nc
                    ...
        """
        logger.info("Preprocessing NSIDC %s daily sea ice drift ...", hemisphere)
        out_file = self.drift_dir / f'NSIDC_PolarPathfinder_sidrift_daily_{hemisphere}_19790101-20231231.nc'
        if out_file.exists():
            logger.info("  -> output already exists, skipping.")
            return
        data_path = Path(data_path)
        temp_folder = self.drift_dir / f'temp_NSIDC_{hemisphere}'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            hemisphere_dir = 'north' if hemisphere == 'nh' else 'south'
            # --- Merge to one file ---
            cdo.mergetime(input=str(data_path / hemisphere_dir / 'daily' / '*.nc'),
                          output=str(temp_folder / 'temp1.nc'))
            cdo.expr("'u=u/100.0;v=v/100.0;icemotion_error_estimate=icemotion_error_estimate/1000.0'",
                     input=str(temp_folder / 'temp1.nc'),
                     output=str(temp_folder / 'temp2.nc'))  # convert from [cm/s] to [m/s]
            # --- Mark values greater than 1 as coastal flag, then subtract 1 to recover actual uncertainty ---
            cdo.aexpr(
                """'error_mask=(icemotion_error_estimate>1)?1:0;
                   icemotion_error_estimate=(icemotion_error_estimate>1)?
                   (icemotion_error_estimate-1):icemotion_error_estimate'""",
                input=str(temp_folder / 'temp2.nc'),
                output=str(temp_folder / 'temp3.nc'))
            cdo.setattribute(
                "u@units='m/s'," +
                "v@units='m/s'," +
                "icemotion_error_estimate@comment='Values between 0 and 1 represent uncertainty (std) in m/s; "
                "Values less than 0 indicate vectors within 25 km of coastline and should be removed in use.'",
                input=str(temp_folder / 'temp3.nc'),
                output=str(temp_folder / 'temp4.nc'))
            # --- Set grid ---
            utils.extract_grid(nc_file=str(temp_folder / 'temp1.nc'),
                               lon_name='longitude', lat_name='latitude',
                               grid_file=str(temp_folder / f'NSIDC_temp_{hemisphere}'))
            cdo.setgrid(str(temp_folder / f'NSIDC_temp_{hemisphere}.txt'),
                        input=str(temp_folder / 'temp4.nc'),
                        output=str(self.drift_dir /
                                   f'NSIDC_PolarPathfinder_sidrift_daily_{hemisphere}_19790101-20231231.nc'))
        finally:
            shutil.rmtree(temp_folder, ignore_errors=True)

        logger.info("NSIDC %s daily drift completed in %.0fs.", hemisphere, time.time() - ts)

    # -----------------------------------------------------------------------------
    # Function: `prep_OSI_SIdrift`.
    # Purpose: Handle prep OSI SIdrift workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    def prep_OSI_SIdrift(self, data_path, hemisphere):
        """Process and merge OSI SAF daily sea ice drift files.

        Args:
            data_path: Directory of /drift_lr folder.
            hemisphere: Target hemisphere: 'nh' or 'sh'.

        Data Info
        ---------
        Avaliable at: https://doi.org/10.15770/EUM_SAF_OSI_0012
        Can be downloaded from OSI SAF FTP server:
            anonymous@osisaf.met.no
        Directory:
            /reprocessed/ice/drift_lr/v1           (1978-2020)

        Directory tree
        --------------
        |--OSISAF
            |--drift_lr
                |--v1
                    |--merged
                        |--1991
                            |--01
                                |--ice_drift_nh_ease2-750_cdr-v1p0_24h-199101011200.nc
                                |--ice_drift_nh_ease2-750_cdr-v1p0_24h-199101021200.nc
                                ...
                                |--ice_drift_sh_ease2-750_cdr-v1p0_24h-199101011200.nc
                            ...
                            |--12
                        ...
                        |--2020
        """
        logger.info("Preprocessing OSI %s daily sea ice drift ...", hemisphere)
        out_file = self.drift_dir / f'OSI-455_sidrift_daily_{hemisphere}_19910101-20201231.nc'
        if out_file.exists():
            logger.info("  -> output already exists, skipping.")
            return
        data_path = Path(data_path)
        temp_folder = self.drift_dir / f'temp_OSI_{hemisphere}'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            # --- Merge to one file ---
            for year in range(1991, 2021):
                files = list((data_path / 'v1' / 'merged' / str(year)).glob(f'**/*_{hemisphere}_*.nc'))
                cdo.mergetime(input=[str(f) for f in files],
                              output=str(temp_folder / f'temp_{hemisphere}_{year}.nc'))
            cdo.mergetime(input=str(temp_folder / f'temp_{hemisphere}_*.nc'),
                          output=str(temp_folder / f'temp_{hemisphere}_1991-2020.nc'))
            # --- Calculate daily uv in x,y direction from raw data ---
            cdo.expr("'u=dX*1000/(t1-t0);v=dY*1000/(t1-t0);unc_ux_vy=uncert_dX_and_dY*1000/(t1-t0)'",
                     input=str(temp_folder / f'temp_{hemisphere}_1991-2020.nc'),
                     output=str(temp_folder / f'temp_{hemisphere}_2.nc'))
            cdo.setattribute(
                "u@units='m/s'," +
                "v@units='m/s'," +
                "unc_ux_vy@comment='Uncertainty (std) in m/s'",
                input=str(temp_folder / f'temp_{hemisphere}_2.nc'),
                output=str(self.drift_dir /
                            f'OSI-455_sidrift_daily_{hemisphere}_19910101-20201231.nc'))
        finally:
            shutil.rmtree(temp_folder, ignore_errors=True)

        logger.info("OSI %s daily drift completed in %.0fs.", hemisphere, time.time() - ts)

    # ======================================================
    #                        SNdepth
    # ======================================================

    def prep_MPMR_SNdepth(self, data_path, grid_file):
        """Process MPMR Antarctic snow depth data from TIFF to NetCDF format.

        Args:
            data_path: Directory containing /MPMR/Antarctic_SnowDepth/ subfolder.
            grid_file: Path to NSIDC0771 lat/lon grid file.

        Data Info
        ---------
        https://doi.org/10.11888/Snow.tpdc.271653
        http://data.tpdc.ac.cn/en/disallow/61ea8177-7177-4507-aeeb-0c7b653d6fc3/ (last access: 2024/11/24)

        Reference:
        Shen, X., Ke, C.-Q., & Li, H. (2022). Snow depth product over Antarctic sea ice from 2002 to 2020
        using multisource passive microwave radiometers. Earth System Science Data, 14(2), 619–636.
        https://doi.org/10.5194/essd-14-619-2022

        Downloaded from National Tibetan Plateau/Third Pole Environment Data Center ftp server:
            host: ftp2.tpdc.ac.cn OR ftp3.tpdc.ac.cn
            port: 6201
            username: download_69736064
            password: 19358359

        Directory tree
        --------------
        |--MPMR
            |--Antarctic_SnowDepth
                |--2002
                    |--20020601_SD.tif
                    |--20020602_SD.tif
                    ...
                ...
                |--2020
            |--Antarctic_SnowDepth_uncertainty
        """
        logger.info("Preprocessing MPMR daily snow depth ...")
        existing = list(self.sndepth_dir.glob('MPMR_SNdepth_sh_*.nc'))
        if existing:
            logger.info("  -> output already exists (%s), skipping.", existing[0].name)
            return
        data_path = Path(data_path)
        temp_folder = self.sndepth_dir / 'temp_MPMR'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            sd_root = data_path / 'MPMR' / 'Antarctic_SnowDepth'
            unc_root = data_path / 'MPMR' / 'Antarctic_SnowDepth_uncertainty'

            # Determine overall date range from all available files
            all_sd_files = sorted(sd_root.glob('**/*_SD.tif'))
            if not all_sd_files:
                raise FileNotFoundError(f"No *_SD.tif files found under {sd_root}")
            all_dates_sorted = sorted(
                datetime.strptime(f.stem.split('_SD')[0], '%Y%m%d')
                for f in all_sd_files
            )
            start_date = all_dates_sorted[0].strftime('%Y%m%d')
            end_date = all_dates_sorted[-1].strftime('%Y%m%d')
            logger.info("Found %d TIFF files from %s to %s",
                        len(all_sd_files), start_date, end_date)

            # Load reference grid lon/lat once
            with xr.open_dataset(str(grid_file)) as ds_grid:
                longitude = np.array(ds_grid['longitude'])
                latitude = np.array(ds_grid['latitude'])

            year_dirs = sorted([d for d in sd_root.iterdir()
                                 if d.is_dir() and d.name.isdigit()])

            # --- Extract CDO grid description before the (possibly parallel) year loop ---
            grid_txt = temp_folder / 'MPMR_grid_temp.txt'
            if not grid_txt.exists():
                _grid_nc = temp_folder / '_grid_extract.nc'
                xr.Dataset({
                    'longitude': (['y', 'x'], longitude),
                    'latitude':  (['y', 'x'], latitude),
                }).to_netcdf(str(_grid_nc))
                utils.extract_grid(nc_file=str(_grid_nc),
                                   lon_name='longitude', lat_name='latitude',
                                   grid_file=str(temp_folder / 'MPMR_grid_temp'))
                _grid_nc.unlink(missing_ok=True)

            # --- Per-year processing (years are independent once the grid file exists) ---
            if self.inner_jobs > 1:
                logger.debug("MPMR: processing %d years with %d threads.",
                             len(year_dirs), self.inner_jobs)
                with ThreadPoolExecutor(max_workers=self.inner_jobs) as pool:
                    futs = {
                        pool.submit(_mpmr_year, str(yd), str(unc_root),
                                    str(temp_folder), longitude, latitude,
                                    str(grid_txt)): yd
                        for yd in year_dirs
                    }
                    yearly_nc_files = []
                    for fut in as_completed(futs):
                        result = fut.result()
                        if result:
                            yearly_nc_files.append(Path(result))
                yearly_nc_files = sorted(yearly_nc_files)
            else:
                yearly_nc_files = []
                for year_dir in year_dirs:
                    logger.debug("Processing MPMR year %s ...", year_dir.name)
                    result = _mpmr_year(str(year_dir), str(unc_root),
                                        str(temp_folder), longitude, latitude,
                                        str(grid_txt))
                    if result:
                        yearly_nc_files.append(Path(result))

            # Merge all yearly files, compute monthly mean, mask and set fill value
            ready_files = [f for f in yearly_nc_files if f.exists()]
            cdo.mergetime(input=[str(f) for f in ready_files],
                          output=str(temp_folder / 'temp_merged.nc'))
            cdo.monmean(input=str(temp_folder / 'temp_merged.nc'),
                        output=str(temp_folder / 'temp_monmean.nc'))
            cdo.setctomiss(0, input=str(temp_folder / 'temp_monmean.nc'),
                           output=str(temp_folder / 'temp_ctomiss.nc'))
            cdo.setmissval(-9999, input=str(temp_folder / 'temp_ctomiss.nc'),
                           output=str(self.sndepth_dir /
                                      f'MPMR_SNdepth_sh_{start_date}-{end_date}_monmean.nc'))
        finally:
            shutil.rmtree(temp_folder, ignore_errors=True)

        logger.info("MPMR daily snow depth completed in %.0fs.", time.time() - ts)

    # -----------------------------------------------------------------------------
    # Function: `prep_CTOHv2_SNdepth_cont`.
    # Purpose: Handle prep CTOHv2 SNdepth cont workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    def prep_CTOHv2_SNdepth_cont(self, data_path, hemisphere):
        """Process CTOH v2 snow depth data (calculated from freeboard difference).

        Args:
            data_path: Directory of /LEGOS folder containing /v2.0/Arctic|Antarctic/.
            hemisphere: Target hemisphere: 'nh' or 'sh'.

        Note
        ----
        Snow depth is derived by subtracting sea_ice_freeboard from total_freeboard.

        Data Info
        ---------
        Sea ice thickness climate data record based on altimetry missions: ERS-1, ERS-2, Envisat and
        CryoSat-2. Available as EASE2-grid monthly maps at 12.5 km resolution, 1994-2023. Includes radar
        freeboard, sea ice freeboard, total freeboard, sea ice thickness, and uncertainty ensemble statistics.

        Downloaded from:
        https://www.legos.omp.eu/ctoh/catalogue/?uuid=c52f78d1-a658-4c56-a66c-459de0d88fb6
        via personal registered user ID.

        References:
        - Bocquet, M., Fleury, S., Rémy, F., & Piras, F. (2024). Arctic and Antarctic Sea Ice Thickness and
          Volume Changes From Observations Between 1994 and 2023. Journal of Geophysical Research: Oceans,
          129(11), e2023JC020848. https://doi.org/10.1029/2023JC020848
        - Bocquet, M. & Fleury, S. (2023). Arctic and Antarctic sea ice thickness climate data record
          (ERS-1, ERS-2, Envisat, CryoSat-2). [dataset]. Odatis.
          https://doi.org/10.6096/ctoh_sit_2023_01

        Directory tree
        --------------
        |--ctoh
            |--v2.0
                |--Arctic
                    |--SIT_nh_199410_199604_ers1_v2.nc
                    |--SIT_nh_199510_200304_ers2_v2.nc
                    |--SIT_nh_200210_201112_env3_v2.nc
                    |--SIT_nh_201011_202304_cs2E_v2.nc
                |--Antarctic
                    |--SIT_sh_199404_199605_ers1_v2.nc
                    |--SIT_sh_199505_200305_ers2_v2.nc
                    |--SIT_sh_200205_201203_env3_v2.nc
                    |--SIT_sh_201012_202306_cs2E_v2.nc
        """
        logger.info("Preprocessing CTOH v2 %s monthly snow depth ...", hemisphere)
        existing = list(self.sndepth_dir.glob(f'CTOH_sndepth_monthly_{hemisphere}_*.nc'))
        if existing:
            logger.info("  -> output already exists (%s), skipping.", existing[0].name)
            return
        data_path = Path(data_path)
        temp_folder = self.sndepth_dir / f'temp_ctohv2_{hemisphere}'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            hemisphere_dir = 'Arctic' if hemisphere == 'nh' else 'Antarctic'
            # SIT_sh_201012_202306_cs2E_v2.nc is known-broken in v2.0 (HDF read error)
            _broken = {'SIT_sh_201012_202306_cs2E_v2.nc'}
            file_list = [f for f in sorted((data_path / 'v2.0' / hemisphere_dir).glob('*.nc'))
                         if f.name not in _broken]

            # --- Set grid ---
            utils.extract_grid(nc_file=str(file_list[0]),
                               lon_name='lon', lat_name='lat',
                               grid_file=str(temp_folder / f'ctoh_temp_{hemisphere}'))

            for file in file_list:
                bn = file.stem
                cdo.setgrid(str(temp_folder / f'ctoh_temp_{hemisphere}.txt'),
                            input=str(file),
                            output=str(temp_folder / f'{bn}_temp1.nc'))
                # --- Calculate SNdepth ---
                cdo.expr(
                    """'snow_depth=total_freeboard-sea_ice_freeboard;
                         STAT_snow_depth_std=sqrt(pow(STAT_total_freeboard_std,2)+pow(STAT_sea_ice_freeboard_std,2));
                         siconc=ice_concentration'""",
                    input=str(temp_folder / f'{bn}_temp1.nc'),
                    output=str(temp_folder / f'{bn}_temp2.nc'))
                cdo.aexpr(
                    """'snow_depth_eff=snow_depth*siconc/100;
                          STAT_snow_depth_eff_std=STAT_snow_depth_std*siconc/100'""",
                    input=str(temp_folder / f'{bn}_temp2.nc'),
                    output=str(temp_folder / f'{bn}_temp3.nc'))

            cdo.mergetime(input=str(temp_folder / '*_temp3.nc'),
                          output=str(temp_folder / 'temp4.nc'))
            cdo.monmean(input=str(temp_folder / 'temp4.nc'),
                        output=str(temp_folder / 'temp5.nc'))

            timestamp = cdo.showtimestamp(input=str(temp_folder / 'temp5.nc'))
            time_sta = timestamp[0].split()[0]
            time_end = timestamp[0].split()[-1]
            start_ym = time_sta[:7].replace('-', '')
            end_ym = time_end[:7].replace('-', '')

            cdo.setctomiss(0, input=str(temp_folder / 'temp5.nc'),
                           output=str(temp_folder / 'temp6.nc'))
            cdo.setmissval(-9999, input=str(temp_folder / 'temp6.nc'),
                           output=str(self.sndepth_dir /
                                      f'CTOH_sndepth_monthly_{hemisphere}_{start_ym}-{end_ym}.nc'))
        finally:
            shutil.rmtree(temp_folder, ignore_errors=True)

        logger.info("CTOH v2 %s monthly snow depth completed in %.0fs.", hemisphere, time.time() - ts)

    # -----------------------------------------------------------------------------
    # Function: `prep_SnowLG_SNdepth`.
    # Purpose: Handle prep SnowLG SNdepth workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    def prep_SnowLG_SNdepth(self, data_path):
        """Process SnowModel-LG snow depth data.

        Args:
            data_path: Directory of /nsidc0758 folder.

        Data Info
        ---------
        Downloaded from NOAA@NSIDC HTTPS File System:
        https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0758_lagrangian_snow_distributions_v01/

        References:
        - Liston, G. E., Itkin, P., Stroeve, J., Tschudi, M., Stewart, J. S., Pedersen, S. H., et al.
          (2020). A Lagrangian Snow-Evolution System for Sea-Ice Applications (SnowModel-LG): Part I—Model
          Description. Journal of Geophysical Research: Oceans, 125(10).
          https://doi.org/10.1029/2019JC015913
        - Liston, G. E., Stroeve, J. & Itkin, P. (2021). Lagrangian Snow Distributions for Sea-Ice
          Applications. (NSIDC-0758, Version 1). [Data Set]. Boulder, Colorado USA. NASA NSIDC DAAC.
          https://doi.org/10.5067/27A0P5M6LZBI. Date Accessed 11-29-2024.

        Spatial Reference System(s):
            NSIDC EASE-Grid North (EPSG:3408)

        Directory tree
        --------------
        |--nsidc0758_lagrangian_snow_distributions_v01
            |--SM_snod_ERA5_01Aug1980-31Jul2021_v01.nc
            |--SM_snod_MERRA2_01Aug1980-31Jul2021_v01.nc
        """
        logger.info("Preprocessing SnowModel-LG daily snow depth ...")
        era5_out = self.sndepth_dir / 'SnowModel-LG_nh_snod_ERA5_v01_19800801-20210731.nc'
        merra2_out = self.sndepth_dir / 'SnowModel-LG_nh_snod_MERRA2_v01_19800801-20210731.nc'
        if era5_out.exists() and merra2_out.exists():
            logger.info("  -> outputs already exist, skipping.")
            return
        data_path = Path(data_path)
        temp_folder = self.sndepth_dir / 'temp_SnowModel-LG'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            # --- compress ---
            cdo.copy(input=str(data_path / 'SM_snod_ERA5_01Aug1980-31Jul2021_v01.nc'),
                     output=str(temp_folder / 'temp1_ERA5.nc'))
            cdo.copy(input=str(data_path / 'SM_snod_MERRA2_01Aug1980-31Jul2021_v01.nc'),
                     output=str(temp_folder / 'temp1_MERRA2.nc'))

            # --- Set grid ---
            proj_3408 = pyproj.Proj("EPSG:3408")
            transformer = pyproj.Transformer.from_proj(proj_3408, proj_3408.to_latlong(), always_xy=True)

            with xr.open_dataset(str(temp_folder / 'temp1_ERA5.nc')) as ds:
                x_range, y_range = np.array(ds['x']), np.array(ds['y'])

            x, y = np.meshgrid(x_range, y_range)
            lon, lat = transformer.transform(x, y)
            nx, ny = lon.shape

            # --- Write into txt ---
            grid_txt = temp_folder / 'EPSG3408.txt'
            logger.debug("Writing grid info to %s ...", grid_txt)
            with open(str(grid_txt), 'w') as f:
                f.write("# CDO description of a curvilinear grid.\n\n")
                f.write("gridtype = curvilinear\n")
                f.write(f"gridsize = {nx * ny:.12g}\n")
                f.write(f"xsize = {ny:.6g}\n")
                f.write(f"ysize = {nx:.6g}\n\n")
                # write longitudes
                f.write("xvals =\n")
                for row in lon:
                    for v in row:
                        f.write(f"{v:.3f}\n")
                # write latitudes
                f.write("yvals =\n")
                for row in lat:
                    for v in row:
                        f.write(f"{v:.3f}\n")

            # The parcel data outputs from the lagrangian SnowLG model has been scaled to
            # grid-cell-averaged snow depth (i.e., effective snow depth, or snow volume per unit).
            for suffix in ['ERA5', 'MERRA2']:
                cdo.setgrid(str(grid_txt),
                            input=str(temp_folder / f'temp1_{suffix}.nc'),
                            output=str(temp_folder / f'temp2_{suffix}.nc'))
                cdo.chname(f"'snod,snow_depth_eff'",
                           input=str(temp_folder / f'temp2_{suffix}.nc'),
                           output=str(self.sndepth_dir /
                                      f'SnowModel-LG_nh_snod_{suffix}_v01_19800801-20210731.nc'))
        finally:
            shutil.rmtree(temp_folder, ignore_errors=True)

        logger.info("SnowModel-LG nh daily snow depth completed in %.0fs.", time.time() - ts)

    # ======================================================
    #                        SIthick
    # ======================================================

    def prep_CSBD_SIthick(self, data_path):
        """Process CSBD (CryoSat-2 Baseline-D) sea ice thickness data.

        Args:
            data_path: Directory containing /Sea_Ice_Thickness/ subfolder.

        Data Info
        ---------
        Downloaded from:
        https://ramadda.data.bas.ac.uk/repository/entry/show?entryid=synth:d8c66670-57ad-44fc-8fef-942a46734ecb:L1NlYV9JY2VfVGhpY2tuZXNz

        References:
        - Landy, J. C., Dawson, G. J., Tsamados, M., Bushuk, M., Stroeve, J. C., Howell, S. E. L., et al.
          (2022). A year-round satellite sea-ice thickness record from CryoSat-2. Nature, 609(7927), 517–522.
          https://doi.org/10.1038/s41586-022-05058-5
        - Landy, J., & Dawson, G. (2022). Year-round Arctic sea ice thickness from CryoSat-2 Baseline-D
          Level 1b observations 2010-2020 (Version 1.0) [Data set]. NERC EDS UK Polar Data Centre.
          https://doi.org/10.5285/d8c66670-57ad-44fc-8fef-942a46734ecb

        Directory tree
        --------------
        |--Sea_Ice_Thickness
            |--ubristol_cryosat2_seaicethickness_nh_80km_v1p7.nc
        """
        logger.info("Preprocessing CSBD biweekly sea ice thickness ...")
        out_file = self.thick_dir / 'CSBD_ubristol_cs2_sit_nh_80km_v1p7_monthly_201011-202007.nc'
        if out_file.exists():
            logger.info("  -> output already exists, skipping.")
            return
        data_path = Path(data_path)
        temp_folder = self.thick_dir / 'temp_CSBD'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            # --- Set grid ---
            input_file = str(data_path / 'Sea_Ice_Thickness' /
                             'ubristol_cryosat2_seaicethickness_nh_80km_v1p7.nc')
            utils.extract_grid(nc_file=input_file,
                               lon_name='Longitude', lat_name='Latitude',
                               grid_file=str(temp_folder / 'CSBD_grid'))
            cdo.setgrid(str(temp_folder / 'CSBD_grid.txt'),
                        input=input_file, output=str(temp_folder / 'temp1.nc'))
            cdo.delname("'Longitude,Latitude'",
                        input=str(temp_folder / 'temp1.nc'),
                        output=str(temp_folder / 'temp2.nc'))
            cdo.monmean(input=str(temp_folder / 'temp2.nc'),
                        output=str(temp_folder / 'temp3.nc'))
            cdo.seldate("'2010-11-01,2020-07-31'",
                        input=str(temp_folder / 'temp3.nc'),
                        output=str(temp_folder / 'temp4.nc'))
            cdo.chname(
                "'Sea_Ice_Thickness,sithick_eff,Sea_Ice_Thickness_Uncertainty,"
                "sithick_eff_unc,Sea_Ice_Concentration,siconc'",
                input=str(temp_folder / 'temp4.nc'),
                output=str(temp_folder / 'temp5.nc'))

            # Create correct time coordinates from known date range
            with xr.open_dataset(str(temp_folder / 'temp5.nc'), decode_times=False) as ds:
                n_times = len(ds.Time)
                start_date = pd.Timestamp('2010-11-01')
                end_date = pd.Timestamp('2020-07-31')
                # Calculate total months
                months = ((end_date.year - start_date.year) * 12 +
                          (end_date.month - start_date.month) + 1)
                if months == n_times:
                    time_coords = (pd.date_range(start=start_date, end=end_date, freq='MS')
                                   + pd.Timedelta(days=14))
                else:
                    logger.warning("Time dimension length (%d) does not match expected (%d).",
                                   n_times, months)
                    time_coords = (pd.date_range(start=start_date, periods=n_times, freq='MS')
                                   + pd.Timedelta(days=14))
                logger.debug("Time coordinates: %s to %s", time_coords[0], time_coords[-1])
                # Assign new time coordinates and rename dimension
                ds = ds.assign_coords(Time=time_coords).rename({'Time': 'time'})
                ds.to_netcdf(str(self.thick_dir /
                                 'CSBD_ubristol_cs2_sit_nh_80km_v1p7_monthly_201011-202007.nc'))
        finally:
            shutil.rmtree(temp_folder, ignore_errors=True)

        logger.info("CSBD nh monthly sea ice thickness completed in %.0fs.", time.time() - ts)

    # -----------------------------------------------------------------------------
    # Function: `prep_CTOHv2_SIthick_cont`.
    # Purpose: Handle prep CTOHv2 SIthick cont workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    def prep_CTOHv2_SIthick_cont(self, data_path, hemisphere):
        """Process CTOH v2 sea ice thickness data, continuous record (1994-2023).

        Args:
            data_path: Directory of /LEGOS folder containing /v2.0/Arctic|Antarctic/.
            hemisphere: Target hemisphere: 'nh' or 'sh'.

        Data Info
        ---------
        Sea ice thickness climate data record based on altimetry missions: ERS-1, ERS-2, Envisat and
        CryoSat-2. Available as EASE2-grid monthly maps at 12.5 km resolution, 1994-2023. Includes radar
        freeboard, sea ice freeboard, total freeboard, sea ice thickness, and uncertainty ensemble statistics.

        Downloaded from:
        https://www.legos.omp.eu/ctoh/catalogue/?uuid=c52f78d1-a658-4c56-a66c-459de0d88fb6
        via personal registered user ID.

        References:
        - Bocquet, M., Fleury, S., Rémy, F., & Piras, F. (2024). Arctic and Antarctic Sea Ice Thickness and
          Volume Changes From Observations Between 1994 and 2023. Journal of Geophysical Research: Oceans,
          129(11), e2023JC020848. https://doi.org/10.1029/2023JC020848
        - Bocquet, M. & Fleury, S. (2023). Arctic and Antarctic sea ice thickness climate data record
          (ERS-1, ERS-2, Envisat, CryoSat-2). [dataset]. Odatis.
          https://doi.org/10.6096/ctoh_sit_2023_01

        Notes
        -----
        In v2.0, SIT_sh_201012_202306_cs2E_v2.nc is broken.

        Directory tree
        --------------
        |--ctoh
            |--v2.0
                |--Arctic
                    |--SIT_nh_199410_199604_ers1_v2.nc
                    |--SIT_nh_199510_200304_ers2_v2.nc
                    |--SIT_nh_200210_201112_env3_v2.nc
                    |--SIT_nh_201011_202304_cs2E_v2.nc
                |--Antarctic
                    |--SIT_sh_199404_199605_ers1_v2.nc
                    |--SIT_sh_199505_200305_ers2_v2.nc
                    |--SIT_sh_200205_201203_env3_v2.nc
                    |--SIT_sh_201012_202306_cs2E_v2.nc
        """
        logger.info("Preprocessing CTOH v2 %s monthly sea ice thickness ...", hemisphere)
        existing = list(self.thick_dir.glob(f'CTOH_sithick_monthly_{hemisphere}_*.nc'))
        if existing:
            logger.info("  -> output already exists (%s), skipping.", existing[0].name)
            return
        data_path = Path(data_path)
        temp_folder = self.thick_dir / f'temp_CTOHv2_{hemisphere}'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            hemisphere_dir = 'Arctic' if hemisphere == 'nh' else 'Antarctic'
            # SIT_sh_201012_202306_cs2E_v2.nc is known-broken in v2.0 (HDF read error)
            _broken = {'SIT_sh_201012_202306_cs2E_v2.nc'}
            file_list = [f for f in sorted((data_path / 'v2.0' / hemisphere_dir).glob('*.nc'))
                         if f.name not in _broken]

            # --- Set grid ---
            utils.extract_grid(nc_file=str(file_list[0]),
                               lon_name='lon', lat_name='lat',
                               grid_file=str(temp_folder / f'ctoh_temp_{hemisphere}'))

            for file in file_list:
                bn = file.stem
                cdo.setgrid(str(temp_folder / f'ctoh_temp_{hemisphere}.txt'),
                            input=str(file),
                            output=str(temp_folder / f'{bn}_temp1.nc'))
                cdo.selname("'sea_ice_thickness,STAT_sea_ice_thickness_std,ice_concentration'",
                            input=str(temp_folder / f'{bn}_temp1.nc'),
                            output=str(temp_folder / f'{bn}_temp2.nc'))
                cdo.chname("'sea_ice_thickness,sithick,STAT_sea_ice_thickness_std,STAT_sithick_std'",
                           input=str(temp_folder / f'{bn}_temp2.nc'),
                           output=str(temp_folder / f'{bn}_temp3.nc'))
                # Convert to equivalent thickness
                cdo.aexpr(
                    """'sithick_eff=sithick*ice_concentration/100;
                          STAT_sithick_eff_std=STAT_sithick_std*ice_concentration/100'""",
                    input=str(temp_folder / f'{bn}_temp3.nc'),
                    output=str(temp_folder / f'{bn}_temp4.nc'))

            cdo.mergetime(input=str(temp_folder / '*_temp4.nc'),
                          output=str(temp_folder / 'temp5.nc'))
            cdo.monmean(input=str(temp_folder / 'temp5.nc'),
                        output=str(temp_folder / 'temp6.nc'))

            timestamp = cdo.showtimestamp(input=str(temp_folder / 'temp6.nc'))
            time_sta = timestamp[0].split()[0]
            time_end = timestamp[0].split()[-1]
            start_ym = time_sta[:7].replace('-', '')
            end_ym = time_end[:7].replace('-', '')

            cdo.setctomiss(0, input=str(temp_folder / 'temp6.nc'),
                           output=str(temp_folder / 'temp7.nc'))
            cdo.setmissval(-9999, input=str(temp_folder / 'temp7.nc'),
                           output=str(self.thick_dir /
                                      f'CTOH_sithick_monthly_{hemisphere}_{start_ym}-{end_ym}.nc'))
        finally:
            shutil.rmtree(temp_folder, ignore_errors=True)

        logger.info("CTOH v2 %s monthly sea ice thickness completed in %.0fs.",
                    hemisphere, time.time() - ts)

    # -----------------------------------------------------------------------------
    # Function: `_read_binary_thickness`.
    # Purpose: Handle read binary thickness workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    def _read_binary_thickness(self, data_path, nx, ny, temp_folder):
        """Read PIOMAS/GIOMAS binary thickness files and write yearly NetCDF files.

        Args:
            data_path: Path to directory containing /Monthly/heff.H* and /grids/.
            nx: Grid x-dimension.
            ny: Grid y-dimension.
            temp_folder: Temporary directory for output files.

        Returns:
            List of yearly NetCDF file paths.
        """
        data_path = Path(data_path)
        binary_files = sorted(data_path.glob('Monthly/heff.H*'))
        yearly_files = []

        for bf in binary_files:
            year = bf.name.split('H')[-1]
            logger.debug("Processing %s", bf)
            with open(str(bf), 'rb') as fh:
                raw = fh.read()
            data = struct.unpack('f' * (len(raw) // 4), raw)

            native_data = np.full((12, nx, ny), np.nan)
            time_coords = []
            for month in range(12):
                start_idx = month * nx * ny
                native_data[month] = np.array(data[start_idx:start_idx + nx * ny]).reshape(nx, ny)
                time_coords.append(np.datetime64(f'{year}-{month + 1:02d}-15'))

            out_nc = temp_folder / f'{year}.nc'
            yearly_files.append(out_nc)
            ds = xr.Dataset(
                {'sithick_eff': (['time', 'x', 'y'], native_data)},
                coords={'time': time_coords}
            )
            ds.to_netcdf(str(out_nc), 'w')

        return yearly_files

    # -----------------------------------------------------------------------------
    # Function: `prep_GIOMAS_SIthick`.
    # Purpose: Handle prep GIOMAS SIthick workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    def prep_GIOMAS_SIthick(self, data_path):
        """Process GIOMAS sea ice thickness data from binary format.

        Args:
            data_path: Directory containing /Monthly/heff.H* and /grids/grid.dat.
        """
        logger.info("Preprocessing GIOMAS monthly sea ice thickness ...")
        out_file = self.thick_dir / 'GIOMAS_sh_sithick_mon_197901-202312.nc'
        if out_file.exists():
            logger.info("  -> output already exists, skipping.")
            return
        data_path = Path(data_path)
        temp_folder = self.thick_dir / 'temp_GIOMAS'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            nx, ny = 276, 360
            raw = np.loadtxt(str(data_path / 'grids' / 'grid.dat'))
            half = len(raw) // 2
            lon = raw[:half].reshape(nx, ny)
            lat = raw[half:].reshape(nx, ny)

            yearly_files = self._read_binary_thickness(data_path, nx, ny, temp_folder)

            # Attach lon/lat and set grid for first file, then all files
            for ii, yf in enumerate(yearly_files):
                with xr.open_dataset(str(yf)) as ds:
                    ds = ds.assign({'longitude': (['x', 'y'], lon),
                                    'latitude': (['x', 'y'], lat)})
                    ds.to_netcdf(str(yf.with_suffix('.tmp.nc')))
                yf.unlink()
                (yf.with_suffix('.tmp.nc')).rename(yf)

                if ii == 0:
                    utils.extract_grid(nc_file=str(yf),
                                       lon_name='longitude', lat_name='latitude',
                                       grid_file=str(temp_folder / 'grid_GIOMAS'))
                cdo.setgrid(str(temp_folder / 'grid_GIOMAS.txt'),
                            input=str(yf),
                            output=str(temp_folder / f'{yf.stem}_grid.nc'))

            cdo.mergetime(input=str(temp_folder / '*_grid.nc'),
                          output=str(temp_folder / 'temp1.nc'))
            cdo.delname("'longitude,latitude'",
                        input=str(temp_folder / 'temp1.nc'),
                        output=str(temp_folder / 'temp2.nc'))
            cdo.setctomiss(0, input=str(temp_folder / 'temp2.nc'),
                           output=str(temp_folder / 'temp3.nc'))
            cdo.setmissval(-9999, input=str(temp_folder / 'temp3.nc'),
                           output=str(self.thick_dir / 'GIOMAS_sh_sithick_mon_197901-202312.nc'))
        finally:
            shutil.rmtree(temp_folder, ignore_errors=True)

        logger.info("GIOMAS monthly sea ice thickness completed in %.0fs.", time.time() - ts)

    # -----------------------------------------------------------------------------
    # Function: `prep_TOPAZ_SIthick`.
    # Purpose: Handle prep TOPAZ SIthick workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    def prep_TOPAZ_SIthick(self, data_path):
        """Process TOPAZ4 sea ice thickness data.

        Args:
            data_path: Directory containing cmems*.nc files.

        Data Info
        ---------
        https://doi.org/10.48670/moi-00007

        Downloaded from:
        https://data.marine.copernicus.eu/product/ARCTIC_MULTIYEAR_PHY_002_003/download?dataset=cmems_mod_arc_phy_my_topaz4_P1D-m_202211
        via personal registered user ID.

        Notes
        -----
        Although TOPAZ4 provides daily sea ice variables, only monthly sea ice thickness was downloaded
        for model evaluation. Only winter sea ice thickness is assimilated (end of October to early April).
        The sea ice thickness output from TOPAZ4 represents gridcell-averaged ice thickness
        (Schweiger et al., 2011).

        References:
        - Xiu, Y., Min, C., Xie, J., Mu, L., Han, B., & Yang, Q. (2021). Evaluation of sea-ice thickness
          reanalysis data from the coupled ocean-sea-ice data assimilation system TOPAZ4. Journal of
          Glaciology, 67(262), 353–365. https://doi.org/10.1017/jog.2020.110

        Directory tree
        --------------
        |--TOPAZ4
            |--cmems_mod_arc_phy_my_topaz4_P1M_1732887619086.nc
            |--cmems_mod_arc_phy_my_topaz4_P1M_1732887773344.nc
            ...
        """
        logger.info("Preprocessing TOPAZ4b monthly sea ice thickness ...")
        out_file = self.thick_dir / 'TOPAZ4b_nh_sithick_mon_199101-202312.nc'
        if out_file.exists():
            logger.info("  -> output already exists, skipping.")
            return
        data_path = Path(data_path)
        temp_folder = self.thick_dir / 'temp_TOPAZ4'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            # --- Merge to one file ---
            cdo.mergetime(input=str(data_path / 'cmems*.nc'),
                          output=str(temp_folder / 'temp1.nc'))
            # --- Intercept variable ---
            cdo.chname("'sithick,sithick_eff'",
                       input=str(temp_folder / 'temp1.nc'),
                       output=str(temp_folder / 'temp2.nc'))
            # --- Set grid ---
            utils.extract_grid(nc_file=str(temp_folder / 'temp2.nc'),
                               lon_name='longitude', lat_name='latitude',
                               grid_file=str(temp_folder / 'TOPAZ4_grid'))
            cdo.setgrid(str(temp_folder / 'TOPAZ4_grid.txt'),
                        input=str(temp_folder / 'temp2.nc'),
                        output=str(temp_folder / 'temp3.nc'))
            cdo.delname("'longitude,latitude'",
                        input=str(temp_folder / 'temp3.nc'),
                        output=str(temp_folder / 'temp4.nc'))
            cdo.setctomiss(0, input=str(temp_folder / 'temp4.nc'),
                           output=str(temp_folder / 'temp5.nc'))
            cdo.setmissval(-9999, input=str(temp_folder / 'temp5.nc'),
                           output=str(self.thick_dir / 'TOPAZ4b_nh_sithick_mon_199101-202312.nc'))
        finally:
            shutil.rmtree(temp_folder, ignore_errors=True)

        logger.info("TOPAZ4b nh monthly sea ice thickness completed in %.0fs.", time.time() - ts)

    # -----------------------------------------------------------------------------
    # Function: `prep_PIOMAS_SIthick`.
    # Purpose: Handle prep PIOMAS SIthick workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    def prep_PIOMAS_SIthick(self, data_path):
        """Process PIOMAS sea ice thickness data from binary format.

        Args:
            data_path: Directory containing /Monthly/heff.H* and /grids/longrid.dat, latgrid.dat.

        Notes
        -----
        These codes are modified from Robbie Mallet's PIOMAS reader/netcdf converter:
        https://github.com/robbiemallett/piomas_bin_reader
        """
        logger.info("Preprocessing PIOMAS monthly sea ice thickness ...")
        out_file = self.thick_dir / 'PIOMAS_nh_sithick_mon_197901-202312.nc'
        if out_file.exists():
            logger.info("  -> output already exists, skipping.")
            return
        data_path = Path(data_path)
        temp_folder = self.thick_dir / 'temp_PIOMAS'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            # --- Read grid from binary file ---
            nx, ny = 120, 360
            lon = np.loadtxt(str(data_path / 'grids' / 'longrid.dat')).reshape(nx, ny)
            lat = np.loadtxt(str(data_path / 'grids' / 'latgrid.dat')).reshape(nx, ny)

            # --- Merge to a .nc file ---
            yearly_files = self._read_binary_thickness(data_path, nx, ny, temp_folder)

            for ii, yf in enumerate(yearly_files):
                with xr.open_dataset(str(yf)) as ds:
                    ds = ds.assign({'longitude': (['x', 'y'], lon),
                                    'latitude': (['x', 'y'], lat)})
                    ds.to_netcdf(str(yf.with_suffix('.tmp.nc')))
                yf.unlink()
                (yf.with_suffix('.tmp.nc')).rename(yf)

                # --- Set grid ---
                if ii == 0:
                    utils.extract_grid(nc_file=str(yf),
                                       lon_name='longitude', lat_name='latitude',
                                       grid_file=str(temp_folder / 'grid_PIOMAS'))
                cdo.setgrid(str(temp_folder / 'grid_PIOMAS.txt'),
                            input=str(yf),
                            output=str(temp_folder / f'{yf.stem}_grid.nc'))

            cdo.mergetime(input=str(temp_folder / '*grid.nc'),
                          output=str(temp_folder / 'temp1.nc'))
            cdo.delname("'longitude,latitude'",
                        input=str(temp_folder / 'temp1.nc'),
                        output=str(temp_folder / 'temp2.nc'))
            cdo.setctomiss(0, input=str(temp_folder / 'temp2.nc'),
                           output=str(temp_folder / 'temp3.nc'))
            cdo.setmissval(-9999, input=str(temp_folder / 'temp3.nc'),
                           output=str(self.thick_dir / 'PIOMAS_nh_sithick_mon_197901-202312.nc'))
        finally:
            shutil.rmtree(temp_folder, ignore_errors=True)

        logger.info("PIOMAS nh monthly sea ice thickness completed in %.0fs.", time.time() - ts)

    # ======================================================
    #                   Other functions
    # ======================================================

    def convert2eff(self, raw_file, variable_names, sic_file, sic_variable_name, output_dir):
        """Convert actual thickness/snow depth to effective thickness (volume per unit area).

        Args:
            raw_file: Path to input thickness/snow depth file.
            variable_names: List of variable names to convert to effective thickness.
            sic_file: Path to sea ice concentration file.
            sic_variable_name: Name of sea ice concentration variable.
            output_dir: Output directory for processed files.
        """
        raw_file = Path(raw_file)
        output_dir = Path(output_dir)
        logger.info("Calculating effective thickness for %s ...", raw_file.name)
        temp_folder = raw_file.parent / 'temp_eff'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            # --- monmean ---
            cdo.monmean(input=str(sic_file),
                        output=str(temp_folder / 'sic_monmean.nc'))

            # --- seldate ---
            with xr.open_dataset(str(raw_file)) as ds, \
                    xr.open_dataset(str(temp_folder / 'sic_monmean.nc')) as ds_sic:
                thickness_times = pd.to_datetime(ds['time'].values)
                sic_start_time = pd.to_datetime(ds_sic['time'].values)[0]
                months_since_start = (
                    (thickness_times.year - sic_start_time.year) * 12 +
                    (thickness_times.month - sic_start_time.month)
                ).astype(int)
                month_indices = ','.join(map(str, months_since_start.tolist()))

            cdo.seltimestep(month_indices,
                            input=str(temp_folder / 'sic_monmean.nc'),
                            output=str(temp_folder / 'sic_matched_times.nc'))
            # --- Interpolate ---
            utils.extract_grid(nc_file=str(raw_file),
                               lon_name='lon', lat_name='lat',
                               grid_file=str(temp_folder / 'thickness_grid'))
            cdo.remapbil(str(temp_folder / 'thickness_grid.txt'),
                         input=str(temp_folder / 'sic_matched_times.nc'),
                         output=str(temp_folder / 'sic_interpolated.nc'))
            # --- Calculate heff ---
            cdo.merge(input=[str(temp_folder / 'sic_interpolated.nc'), str(raw_file)],
                      output=str(temp_folder / 'merged_datasets.nc'))

            expr = ';'.join(f'{v}_eff={v}*{sic_variable_name}/100' for v in variable_names)
            cdo.aexpr(f"'{expr}'",
                      input=str(temp_folder / 'merged_datasets.nc'),
                      output=str(temp_folder / 'effective_thickness_calculated.nc'))

            # --- Use the same mask ---
            with xr.open_dataset(str(temp_folder / 'effective_thickness_calculated.nc')) as ds:
                for v in variable_names:
                    ds[f'{v}_eff'] = ds[f'{v}_eff'].where(~ds[v].isnull(), np.nan)
                encoding = {var: {'zlib': True, 'complevel': 5} for var in ds.data_vars}
                ds.attrs = {'remarks': (
                    "The effective thickness (i.e., volume per unit) is calculated from the actual "
                    "thickness multiplied by the density of the NSIDC CDR (ID:G02202_V5) sea ice "
                    "concentration."
                )}
                raw_file.unlink()
                ds.to_netcdf(str(temp_folder / 'temp.nc'), encoding=encoding)

            cdo.setmissval(-9999,
                           input=str(temp_folder / 'temp.nc'),
                           output=str(output_dir / raw_file.name))
        finally:
            shutil.rmtree(temp_folder, ignore_errors=True)

        logger.info("Effective thickness calculation for %s completed in %.0fs.",
                    raw_file.name, time.time() - ts)

if __name__ == '__main__':
    pass
