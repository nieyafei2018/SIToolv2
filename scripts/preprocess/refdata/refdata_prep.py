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
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from scripts import utils

logger = logging.getLogger(__name__)
cdo = Cdo(options=['-f nc4 -z zip'])


class ReferenceDataManager:
    """Manages preprocessing of sea ice reference datasets.

    Args:
        ref_data_dir: Root directory for processed reference data output.
    """

    def __init__(self, ref_data_dir):
        self.ref_data_dir = Path(ref_data_dir)
        self.sic_dir = self.ref_data_dir / 'SIconc'
        self.drift_dir = self.ref_data_dir / 'SIdrift'
        self.sndepth_dir = self.ref_data_dir / 'SNdepth'
        self.thick_dir = self.ref_data_dir / 'SIthick'
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
        """
        logger.info("Preprocessing NSIDC %s daily SIC ...", hemisphere)
        data_path = Path(data_path)
        temp_folder = self.sic_dir / f'temp_NSIDC_{hemisphere}'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            hemisphere_dir = 'north' if hemisphere == 'nh' else 'south'
            for file in data_path.glob(f'{hemisphere_dir}/aggregate/*.nc'):
                cdo.selvar("'cdr_seaice_conc,cdr_seaice_conc_stdev'",
                           input=str(file),
                           output=str(temp_folder / file.name))

            cdo.mergetime(input=str(temp_folder / '*.nc'),
                          output=str(temp_folder / 'temp1.nc'))
            cdo.expr("'siconc=cdr_seaice_conc*100;siconc_std=cdr_seaice_conc_stdev*100'",
                     input=str(temp_folder / 'temp1.nc'),
                     output=str(temp_folder / 'temp2.nc'))

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

    def prep_OSI_SIconc(self, data_path, hemisphere):
        """Process and merge OSI-450 daily sea ice concentration files.

        Args:
            data_path: Directory of /v3p1 folder.
            hemisphere: Target hemisphere: 'nh' or 'sh'.
        """
        logger.info("Preprocessing OSI %s daily SIC ...", hemisphere)
        data_path = Path(data_path)
        temp_folder = self.sic_dir / f'temp_OSI_{hemisphere}'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            for year in range(1979, 2025):
                logger.debug("Processing year: %d", year)
                file_list = list(data_path.glob(f'{year}/*/*{hemisphere}*'))
                out_year = temp_folder / f'temp1_{year}.nc'
                if file_list and not out_year.exists():
                    cdo.mergetime(input=[str(f) for f in file_list],
                                  output=str(out_year))
                    cdo.selname("'ice_conc,total_standard_uncertainty'",
                                input=str(out_year),
                                output=str(temp_folder / f'temp1_{year}_ice.nc'))

            cdo.mergetime(input=str(temp_folder / 'temp1_*_ice.nc'),
                          output=str(temp_folder / 'temp1.nc'))

            sample_file = next(data_path.glob(f'*/*/{hemisphere}*'))
            utils.extract_grid(nc_file=str(sample_file),
                               lon_name='lon', lat_name='lat',
                               grid_file=str(temp_folder / f'OSI_CDR_{hemisphere}'))
            cdo.setgrid(str(temp_folder / f'OSI_CDR_{hemisphere}.txt'),
                        input=str(temp_folder / 'temp1.nc'),
                        output=str(temp_folder / 'temp2.nc'))
            cdo.chname("'ice_conc,siconc,total_standard_uncertainty,siconc_std'",
                       input=str(temp_folder / 'temp2.nc'),
                       output=str(temp_folder / 'temp3.nc'))
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
        """
        logger.info("Preprocessing NSIDC %s daily sea ice drift ...", hemisphere)
        data_path = Path(data_path)
        temp_folder = self.drift_dir / f'temp_NSIDC_{hemisphere}'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            hemisphere_dir = 'north' if hemisphere == 'nh' else 'south'
            cdo.mergetime(input=str(data_path / hemisphere_dir / 'daily' / '*.nc'),
                          output=str(temp_folder / 'temp1.nc'))
            cdo.expr("'u=u/100.0;v=v/100.0;icemotion_error_estimate=icemotion_error_estimate/1000.0'",
                     input=str(temp_folder / 'temp1.nc'),
                     output=str(temp_folder / 'temp2.nc'))
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

    def prep_OSI_SIdrift(self, data_path, hemisphere):
        """Process and merge OSI SAF daily sea ice drift files.

        Args:
            data_path: Directory of /drift_lr folder.
            hemisphere: Target hemisphere: 'nh' or 'sh'.
        """
        logger.info("Preprocessing OSI %s daily sea ice drift ...", hemisphere)
        data_path = Path(data_path)
        temp_folder = self.drift_dir / f'temp_OSI_{hemisphere}'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            for year in range(1991, 2021):
                files = list((data_path / 'v1' / 'merged' / str(year)).glob(f'**/*_{hemisphere}_*.nc'))
                cdo.mergetime(input=[str(f) for f in files],
                              output=str(temp_folder / f'temp_{hemisphere}_{year}.nc'))
            cdo.mergetime(input=str(temp_folder / f'temp_{hemisphere}_*.nc'),
                          output=str(temp_folder / f'temp_{hemisphere}_1991-2020.nc'))
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
        """
        logger.info("Preprocessing MPMR daily snow depth ...")
        data_path = Path(data_path)
        temp_folder = self.sndepth_dir / 'temp_MPMR'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            tiff_files, dates = [], []
            for f in (data_path / 'MPMR' / 'Antarctic_SnowDepth').glob('**/*_SD.tif'):
                dates.append(datetime.strptime(f.stem.split('_SD')[0], '%Y%m%d'))
                tiff_files.append(f)

            sorted_indices = sorted(range(len(dates)), key=lambda i: dates[i])
            tiff_files = [tiff_files[i] for i in sorted_indices]
            dates = [dates[i] for i in sorted_indices]
            unc_files = [Path(str(f).replace('Antarctic_SnowDepth', 'Antarctic_SnowDepth_uncertainty')
                              .replace('_SD.tif', '_SDunc.tif')) for f in tiff_files]

            start_date = dates[0].strftime('%Y%m%d')
            end_date = dates[-1].strftime('%Y%m%d')
            logger.info("Found %d TIFF files from %s to %s", len(tiff_files), start_date, end_date)

            snow_depth_data = []
            for f in tqdm(tiff_files, desc="Reading snow depth TIFF files"):
                with rasterio.open(str(f)) as src:
                    snow_depth_data.append(src.read(1))
            snow_depth_data = np.array(snow_depth_data) / 100

            snow_depth_unc = []
            for f in tqdm(unc_files, desc="Reading uncertainty TIFF files"):
                with rasterio.open(str(f)) as src:
                    snow_depth_unc.append(src.read(1))
            snow_depth_unc = np.array(snow_depth_unc) / 100

            with xr.open_dataset(str(grid_file)) as ds_grid:
                longitude = np.array(ds_grid['longitude'])
                latitude = np.array(ds_grid['latitude'])

            ds_sd = xr.Dataset({
                'snow_depth': (['time', 'y', 'x'], snow_depth_data),
                'snow_depth_unc': (['time', 'y', 'x'], snow_depth_unc),
                'longitude': (['y', 'x'], longitude),
                'latitude': (['y', 'x'], latitude),
            }, coords={'time': dates})
            encoding = {v: {'zlib': True, 'complevel': 4} for v in ds_sd.data_vars}
            ds_sd.to_netcdf(str(temp_folder / 'temp1.nc'), encoding=encoding)

            utils.extract_grid(nc_file=str(temp_folder / 'temp1.nc'),
                               lon_name='longitude', lat_name='latitude',
                               grid_file=str(temp_folder / 'MPMR_grid_temp'))
            cdo.setgrid(str(temp_folder / 'MPMR_grid_temp.txt'),
                        input=str(temp_folder / 'temp1.nc'),
                        output=str(temp_folder / 'temp2.nc'))
            cdo.delname("'longitude,latitude'",
                        input=str(temp_folder / 'temp2.nc'),
                        output=str(temp_folder / 'temp3.nc'))
            cdo.monmean(input=str(temp_folder / 'temp3.nc'),
                        output=str(temp_folder / 'temp4.nc'))
            cdo.setctomiss(0, input=str(temp_folder / 'temp4.nc'),
                           output=str(temp_folder / 'temp5.nc'))
            cdo.setmissval(-9999, input=str(temp_folder / 'temp5.nc'),
                           output=str(self.sndepth_dir /
                                      f'MPMR_SNdepth_sh_{start_date}-{end_date}_monmean.nc'))
        finally:
            shutil.rmtree(temp_folder, ignore_errors=True)

        logger.info("MPMR daily snow depth completed in %.0fs.", time.time() - ts)

    def prep_CTOHv2_SNdepth_cont(self, data_path, hemisphere):
        """Process CTOH v2 snow depth data (calculated from freeboard difference).

        Args:
            data_path: Directory of /LEGOS folder containing /v2.0/Arctic|Antarctic/.
            hemisphere: Target hemisphere: 'nh' or 'sh'.
        """
        logger.info("Preprocessing CTOH v2 %s monthly snow depth ...", hemisphere)
        data_path = Path(data_path)
        temp_folder = self.sndepth_dir / f'temp_ctohv2_{hemisphere}'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            hemisphere_dir = 'Arctic' if hemisphere == 'nh' else 'Antarctic'
            file_list = list((data_path / 'v2.0' / hemisphere_dir).glob('*.nc'))

            utils.extract_grid(nc_file=str(file_list[0]),
                               lon_name='lon', lat_name='lat',
                               grid_file=str(temp_folder / f'ctoh_temp_{hemisphere}'))

            for file in file_list:
                bn = file.stem
                cdo.setgrid(str(temp_folder / f'ctoh_temp_{hemisphere}.txt'),
                            input=str(file),
                            output=str(temp_folder / f'{bn}_temp1.nc'))
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

    def prep_SnowLG_SNdepth(self, data_path):
        """Process SnowModel-LG snow depth data.

        Args:
            data_path: Directory of /nsidc0758 folder.
        """
        logger.info("Preprocessing SnowModel-LG daily snow depth ...")
        data_path = Path(data_path)
        temp_folder = self.sndepth_dir / 'temp_SnowModel-LG'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            cdo.copy(input=str(data_path / 'SM_snod_ERA5_01Aug1980-31Jul2021_v01.nc'),
                     output=str(temp_folder / 'temp1_ERA5.nc'))
            cdo.copy(input=str(data_path / 'SM_snod_MERRA2_01Aug1980-31Jul2021_v01.nc'),
                     output=str(temp_folder / 'temp1_MERRA2.nc'))

            proj_3408 = pyproj.Proj("EPSG:3408")
            transformer = pyproj.Transformer.from_proj(proj_3408, proj_3408.to_latlong(), always_xy=True)

            with xr.open_dataset(str(temp_folder / 'temp1_ERA5.nc')) as ds:
                x_range, y_range = np.array(ds['x']), np.array(ds['y'])

            x, y = np.meshgrid(x_range, y_range)
            lon, lat = transformer.transform(x, y)
            nx, ny = lon.shape

            grid_txt = temp_folder / 'EPSG3408.txt'
            logger.debug("Writing grid info to %s ...", grid_txt)
            with open(str(grid_txt), 'w') as f:
                f.write("# CDO description of a curvilinear grid.\n\n")
                f.write("gridtype = curvilinear\n")
                f.write(f"gridsize = {nx * ny:.12g}\n")
                f.write(f"xsize = {ny:.6g}\n")
                f.write(f"ysize = {nx:.6g}\n\n")
                f.write("xvals =\n")
                for row in lon:
                    for v in row:
                        f.write(f"{v:.3f}\n")
                f.write("yvals =\n")
                for row in lat:
                    for v in row:
                        f.write(f"{v:.3f}\n")

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
        """
        logger.info("Preprocessing CSBD biweekly sea ice thickness ...")
        data_path = Path(data_path)
        temp_folder = self.thick_dir / 'temp_CSBD'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
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

    def prep_CTOHv2_SIthick_cont(self, data_path, hemisphere):
        """Process CTOH v2 sea ice thickness data, continuous record (1994-2023).

        Args:
            data_path: Directory of /LEGOS folder containing /v2.0/Arctic|Antarctic/.
            hemisphere: Target hemisphere: 'nh' or 'sh'.
        """
        logger.info("Preprocessing CTOH v2 %s monthly sea ice thickness ...", hemisphere)
        data_path = Path(data_path)
        temp_folder = self.thick_dir / f'temp_CTOHv2_{hemisphere}'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            hemisphere_dir = 'Arctic' if hemisphere == 'nh' else 'Antarctic'
            file_list = list((data_path / 'v2.0' / hemisphere_dir).glob('*.nc'))

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

        for ii, bf in enumerate(binary_files):
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

    def prep_GIOMAS_SIthick(self, data_path):
        """Process GIOMAS sea ice thickness data from binary format.

        Args:
            data_path: Directory containing /Monthly/heff.H* and /grids/grid.dat.
        """
        logger.info("Preprocessing GIOMAS monthly sea ice thickness ...")
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

    def prep_TOPAZ_SIthick(self, data_path):
        """Process TOPAZ4 sea ice thickness data.

        Args:
            data_path: Directory containing cmems*.nc files.
        """
        logger.info("Preprocessing TOPAZ4b monthly sea ice thickness ...")
        data_path = Path(data_path)
        temp_folder = self.thick_dir / 'temp_TOPAZ4'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            cdo.mergetime(input=str(data_path / 'cmems*.nc'),
                          output=str(temp_folder / 'temp1.nc'))
            cdo.chname("'sithick,sithick_eff'",
                       input=str(temp_folder / 'temp1.nc'),
                       output=str(temp_folder / 'temp2.nc'))
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

    def prep_PIOMAS_SIthick(self, data_path):
        """Process PIOMAS sea ice thickness data from binary format.

        Args:
            data_path: Directory containing /Monthly/heff.H* and /grids/longrid.dat, latgrid.dat.
        """
        logger.info("Preprocessing PIOMAS monthly sea ice thickness ...")
        data_path = Path(data_path)
        temp_folder = self.thick_dir / 'temp_PIOMAS'
        temp_folder.mkdir(parents=True, exist_ok=True)
        ts = time.time()

        try:
            nx, ny = 120, 360
            lon = np.loadtxt(str(data_path / 'grids' / 'longrid.dat')).reshape(nx, ny)
            lat = np.loadtxt(str(data_path / 'grids' / 'latgrid.dat')).reshape(nx, ny)

            yearly_files = self._read_binary_thickness(data_path, nx, ny, temp_folder)

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
            cdo.monmean(input=str(sic_file),
                        output=str(temp_folder / 'sic_monmean.nc'))

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
            utils.extract_grid(nc_file=str(raw_file),
                               lon_name='lon', lat_name='lat',
                               grid_file=str(temp_folder / 'thickness_grid'))
            cdo.remapbil(str(temp_folder / 'thickness_grid.txt'),
                         input=str(temp_folder / 'sic_matched_times.nc'),
                         output=str(temp_folder / 'sic_interpolated.nc'))
            cdo.merge(input=[str(temp_folder / 'sic_interpolated.nc'), str(raw_file)],
                      output=str(temp_folder / 'merged_datasets.nc'))

            expr = ';'.join(f'{v}_eff={v}*{sic_variable_name}/100' for v in variable_names)
            cdo.aexpr(f"'{expr}'",
                      input=str(temp_folder / 'merged_datasets.nc'),
                      output=str(temp_folder / 'effective_thickness_calculated.nc'))

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
