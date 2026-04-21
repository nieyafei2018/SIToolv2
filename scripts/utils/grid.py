# -*- coding: utf-8 -*-
"""Split utility helpers."""

import os

from scripts.utils import core as _core

# Reuse shared utility namespace and constants from core module.
globals().update({k: v for k, v in _core.__dict__.items() if k not in globals()})


def _runtime_tmpdir() -> str:
    """Resolve active temporary directory with case-level override."""
    preferred = str(os.environ.get('SITOOL_CASE_TMPDIR', '')).strip()
    if not preferred:
        preferred = str(os.environ.get('SITOOL_TMPDIR', '')).strip()
    if not preferred:
        preferred = str(SITOOL_TMPDIR)
    resolved = os.path.abspath(preferred)
    os.makedirs(resolved, exist_ok=True)
    return resolved


def gen_polar_grid(hms: str, res: float, nx: int = None, ny: int = None, lat_range: list = None,
                   lon_range: list = None, proj_name: str = None, grid_path: str = './',
                   etopo_file: str = None, region_mask_file: str = None, ellps: str = 'WGS84') -> str:
    """
    Generate polar stereographic or Lambert azimuthal equal area projection grid. Both of them have the
    x and y direction look something like these ↓

    ^
    ^                             ,           .        (,,,,         , %,,,,,
    ^                            ,%,,&(       .   #,,,,,           / ,,,,,,,,
    |                          @*,,,,,,%      .,,,,,,*,,*/,,,#/       /,,,,,,
    |           ,.@* .   .**%,,,,,,,,,,,,%#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
    |           ,,,,,,,,,,,,,,,,,,,,,,,,,@    (,,,,,,,,,,,,,,,,,,,%,,,,,,,,,,
    |           ,,,,,,,,,,,,,,,,,,,,,,,,(     .      ,,,,,,,,,,,,%,,,,,,,,,,,
    |           ,,,,,,,,,,,,,,,,,,(           .        #,,,,,,,,,,%,,,@,,,,,,
    |           ,,,,,@,,,,&(,,,&              .         ,@,,,,,,,@,,@,,,,,,,,
    |           ,,,,,%,,,,(,,,  .             .       ,,   (,,,,,,,,,,,,,,,,,
    |           ,,,,,,,,,,&*,(&*,,                          %,,,,,,,,,,,,,,,,
    |           ,,,,,,,,,,.,,*#  ,&@  .                   ,,.,,,,,,,,,,,,,,,,
y   |           *,,,,,,,,(,. ,% ., /(                   . ,,,,,,,,,,,,,,,,,,,
    |               @,,,,,#   @ ,@ &,,#                     ,,,,,,,,,,,,,,,,,
    |               &,((,,,(,,, ,&(,@,,,                      ,#&@,,,,,,,,,,,
    |                     ##,@    @&,,,,%&,               (    ,,,%%,,,,,,,,&
    |           ,,,,( *,,,,@*     #,,,,,,,/*              ,    /,%,%,,,,,@,,,
    |           ,,,, /@, @@     &,,,,,,,,,,   . @&.        &(. &,,,%,,,,&/,,,
    |           ,,@(.          %,,,,,,,,,@    .                #,,,,,,,,,,,,,
    |           ,,&        @,,,,,,,,,,,&%     .              (#,,,,,,,,,,,,,,
    |           /         ,,,,,,,,,,,#%@      .       @,,/,,,#,,,,,,,,,,,,,,,
    |           %         @,(                 .     @,,,,,,, %,,,,,,,,,,,,,,,
    |                               #%/*@     .    *,,,.,,,,,@(,,,,,,,,@,,,,,
    |                                         .    (,,, ,,,@,.,,,,,,,,,,,,,,,
    |     .                            .  @,,,,,,  *,,,,,,,,,,,,,,,,,,
    |                                  .  &,%,,,   ,/,,,,,,,,,,,,,,,,,
    |
    ————————————————————————————————————————————————————————————————————————————————>>>
                                             x

    ^
    ^
    ^
    |
    |
    |
    |
    |
    |
    |                                   ,,,,,,,,,,
    |                                 &,,,,,,,,,,,,,,,,,,#
    |                   #             ,,,,,,,,,,,,,,,,,,,,,*
y   |                    #,,,#       %,,,,,,,,,,,,,,,,,,,,,,%
    |                      ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,%
    |                        /,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,&
    |                        ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
    |                       ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,&
    |                          ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,*
    |                           ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,&
    |                               ,, #  %,,,,,,,,,,,,,,,,,%
    |                                          ,,,,,,,,,,,,
    |                                         ,,,,,,,,,,,(
    |
    |
    |     .
    |
    |
    ————————————————————————————————————————————————————————————————————————————————>>>
                                             x

    Parameters
    ----------
    hms: <str>
        which hemisphere, "sh" or "nh"
    proj_name: <str>
        polar stereographic ('stere') or Lambert azimuthal equal area ('laea')
        or ESPG projections (e.g., 'ESPG:6931')
    res: <float>
        horizontal resolution, unit [km]
    nx, ny: <int>
        grid numbers in x and y direction
    grid_path: <str>
        path for storing the output grid files (.txt and .nc)
    etopo_file: <str>
        .nc file downloaded from
        https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc
    region_mask_file: <str>
        .nc file downloaded from https://doi.org/10.5067/CYW3O8ZUNIWC

    Returns
    -------
    (1) A .txt file used for CDO interpolation
    (2) A .nc file which contains the following variables:
        lon, lat, lon_bnds, lat_bnds,
        x_c, y_c, x_bnds, y_bnds,
        cell_area, land_sea_mask, sea_ice_region
    (3) grid_filename : str
        Base name of the grid file (without suffix).

    Some common used projections:
    NSIDC Sea Ice Polar Stereographic North (EPSG:3411)
    NSIDC Sea Ice Polar Stereographic South (EPSG:3412)
        Refer to https://nsidc.org/data/user-resources/help-center/guide-nsidcs-polar-stereographic-projection
    NSIDC EASE-Grid North (EPSG:3408)
    NSIDC EASE-Grid South (EPSG:3409)
    WGS 84 / NSIDC EASE-Grid 2.0 North (EPSG:6931)
    WGS 84 / NSIDC EASE-Grid 2.0 South (EPSG:6932)
        Refer to https://nsidc.org/data/user-resources/help-center/guide-ease-grids

    Usage
    -----
    (1) To generate a 25 km Polar Stereographic South grid with landmask:
    >> Gen_Grid(hms='sh', res=25, nx=320, ny=320, proj_name='stere', grid_path='./',
                landmask=True, etopo_file='/path/to/etopo_file/etopo.nc')

    (2) To reproduce NSIDC EASE-Grid 2.0 North grid without landmask nor region mask:
    >> Gen_Grid(hms='nh', res=25, nx=720, ny=720, proj_name='EPSG:6931', grid_path='./')

    (3) To generate a 60 km Polar Stereographic South grid with landmask and region mask:
    >> Gen_Grid(hms='sh', res=60, nx=144, ny=144, proj_name='stere', grid_path='./',
                etopo_file='../Data/Topography/ETOPO_2022_v1_60s_N90W180_surface.nc',
                region_mask_file='../grids/NSIDC-0780_SeaIceRegions_PS-S3.125km_v1.0.nc',
                ellps='WGS84')

    Notes
    -----
    (1) The EPSG projection (e.g., EPSG:6931) generated by this script is exactly the same as the x, y, latitude and
        longitude given by NSIDC, and there is a negligible error of about 0.1% compared to the original grid area
        provided by NSIDC.
    (2) For EASE grids, using both the grid cell area calculated by this script or simply res**2 are ok.

    """
    # Check if the files used for masking are provided
    if etopo_file is None:
        logger.warning("etopo_file not provided — land mask will be omitted.")
    if region_mask_file is None:
        logger.warning("region_mask_file not provided — region mask will be omitted.")

    grid_filename = f'{proj_name.replace(":", "")}_{hms}_{res}'
    if os.path.exists(f'{grid_path}/{grid_filename}.nc'):
        return f'{grid_path}/{grid_filename}.nc'

    logger.info("-" * 20)
    logger.info("Generating grid ...")
    logger.info("-" * 20)

    # Define projections
    if proj_name in ['stere', 'laea']:
        if hms == 'sh':
            proj = pyproj.Proj(width=8200000, height=8200000, proj=proj_name, lat_0=-90, lon_0=0, ellps=ellps)
            lat_sta, lat_end = -45, -90
        elif hms == 'nh':
            proj = pyproj.Proj(width=7200000, height=7200000, proj=proj_name, lat_0=90, lon_0=0, ellps=ellps)
            lat_sta, lat_end = 45, 90
        else:
            raise ValueError('The hemisphere must be "sh" (Southern) or "nh" (Northern)')
    elif "EPSG" in proj_name:
        proj = pyproj.Proj(proj_name)
    elif proj_name == "lonlat":
        pass
    else:
        raise ValueError('Supported projections: stere, laea, EPSG:XXXX, or lonlat')

    # Generate grid coordinates and cell area
    if (proj_name in ['stere', 'laea']) | ("EPSG" in proj_name):
        if hms == 'sh':
            lat_sta, lat_end = -45, -90
            if (nx is None) | (ny is None):
                nx = ny = int(8400. / res)  # Default domain: 8400 km X 8400 km
        elif hms == 'nh':
            lat_sta, lat_end = 45, 90
            if (nx is None) | (ny is None):
                nx = ny = int(7000. / res)  # Default domain: 7200 km X 7200 km

        # Grid parameters
        dx = dy = res * 1000  # Spatial resolution in meters
        x_min, x_max = -dx * (nx / 2 - 0.5), dx * (nx / 2 - 0.5)
        y_min, y_max = -dy * (ny / 2 - 0.5), dy * (ny / 2 - 0.5)
        delta_x = dx / 2
        delta_y = dy / 2

        # Generate grid coordinates
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        xv, yv = np.meshgrid(x, y)
        xv, yv = xv.T, yv.T  # Rotate to real x and y direction

        # Calculate grid vertex coordinates
        x_bnds = np.zeros((nx, ny, 4))
        y_bnds = np.zeros((nx, ny, 4))
        for i in range(nx):
            for j in range(ny):
                x_bnds[i, j, :] = [xv[i, j] - delta_x, xv[i, j] + delta_x,
                                   xv[i, j] + delta_x, xv[i, j] - delta_x]
                y_bnds[i, j, :] = [yv[i, j] + delta_y, yv[i, j] + delta_y,
                                   yv[i, j] - delta_y, yv[i, j] - delta_y]

        # Convert to geographic coordinates
        lon, lat = proj(xv, yv, inverse=True)
        lon_bnds, lat_bnds = proj(x_bnds, y_bnds, inverse=True)

    elif proj_name == 'lonlat':
        lat_sta, lat_end = lat_range
        lon_sta, lon_end = lon_range

        # Generate grid boundary points
        lat_edges = np.arange(lat_sta, lat_end + res, res)
        lon_edges = np.arange(lon_sta, lon_end + res, res)

        # Grid center coordinates
        lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
        lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
        lon, lat = np.meshgrid(lon_centers, lat_centers)

        # Generate grid vertices
        nx, ny = lon.shape
        lon_bnds = np.zeros((nx, ny, 4))
        lat_bnds = np.zeros((nx, ny, 4))

        for i in range(nx):
            for j in range(ny):
                lon_bnds[i, j, :] = [lon_edges[j], lon_edges[j + 1], lon_edges[j + 1], lon_edges[j]]
                lat_bnds[i, j, :] = [lat_edges[i + 1], lat_edges[i + 1], lat_edges[i], lat_edges[i]]

        xv, yv = np.full((nx, ny), np.nan), np.full((nx, ny), np.nan)
        x_bnds, y_bnds = np.full((nx, ny, 4), np.nan), np.full((nx, ny, 4), np.nan)

    # Calculate grid cell area
    print('\tComputing grid cell area ...')
    areas = _calculate_cell_areas(lon_bnds, lat_bnds, nx, ny)

    # Write grid files
    _write_grid_files(lon, lat, lon_bnds, lat_bnds, xv, yv, x_bnds, y_bnds, areas,
                      nx, ny, grid_path, grid_filename, etopo_file, region_mask_file,
                      lat_sta, lat_end)

    return f'{grid_path}{grid_filename}.nc'

def _calculate_cell_areas(lon_bnds: np.ndarray, lat_bnds: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Approximate spherical quadrilateral grid-cell areas using the Bretschneider formula.

    Method: compute the great-circle distances d1–d4 of the four sides via
    pyproj.Geod, then estimate the area as
        area ≈ √((s-d1)(s-d2)(s-d3)(s-d4))
    where s = (d1+d2+d3+d4)/2 is the semi-perimeter.

    Boundary handling: interior cells (1:nx, 1:ny) are computed normally;
    edge rows/columns are filled by copying the nearest interior value.

    Args:
        lon_bnds: Longitude boundary coordinates (nx, ny, 4).
        lat_bnds: Latitude boundary coordinates (nx, ny, 4).
        nx, ny:   Grid dimensions.

    Returns:
        Area array (nx, ny), units m².
    """

    def haversine_area(lat1, lon1, lat2, lon2, lat3, lon3, lat4, lon4):
        """Estimate spherical quadrilateral area via the Bretschneider formula (units: m²)."""
        geod = pyproj.Geod(ellps='WGS84')
        # Compute great-circle distances for the four sides
        _, _, d1 = geod.inv(lon1, lat1, lon2, lat2)
        _, _, d2 = geod.inv(lon2, lat2, lon3, lat3)
        _, _, d3 = geod.inv(lon3, lat3, lon4, lat4)
        _, _, d4 = geod.inv(lon4, lat4, lon1, lat1)
        s1 = (d1 + d2 + d3 + d4) / 2
        return np.sqrt((s1 - d1) * (s1 - d2) * (s1 - d3) * (s1 - d4))

    # Compute interior cell areas (excluding the outermost boundary ring)
    areas = np.zeros((nx - 1, ny - 1))
    for i in range(nx - 1):
        for j in range(ny - 1):
            lon1, lat1 = lon_bnds[i, j, 0], lat_bnds[i, j, 0]
            lon2, lat2 = lon_bnds[i, j, 1], lat_bnds[i, j, 1]
            lon3, lat3 = lon_bnds[i, j, 2], lat_bnds[i, j, 2]
            lon4, lat4 = lon_bnds[i, j, 3], lat_bnds[i, j, 3]
            areas[i, j] = haversine_area(lat1, lon1, lat2, lon2, lat3, lon3, lat4, lon4)

    # Fill boundary rows/columns with the nearest interior value (avoids NaN or zero)
    area = np.zeros((nx, ny))
    area[1:nx, 1:ny] = areas
    area[0, :], area[-1, :], area[:, 0], area[:, -1] = area[1, :], area[-2, :], area[:, 1], area[:, -2]

    return area

def _write_grid_files(lon: np.ndarray, lat: np.ndarray, lon_bnds: np.ndarray,
                      lat_bnds: np.ndarray, xv: np.ndarray, yv: np.ndarray,
                      x_bnds: np.ndarray, y_bnds: np.ndarray, area: np.ndarray,
                      nx: int, ny: int, grid_path: str, grid_filename: str,
                      etopo_file: str, region_mask_file: str, lat_sta: float,
                      lat_end: float) -> None:
    """Write grid information to text and NetCDF files.

    Args:
        lon: Longitude coordinates
        lat: Latitude coordinates
        lon_bnds: Longitude boundaries
        lat_bnds: Latitude boundaries
        xv: X coordinates
        yv: Y coordinates
        x_bnds: X boundaries
        y_bnds: Y boundaries
        area: Cell areas
        nx: Number of grid points in x direction
        ny: Number of grid points in y direction
        grid_path: Output directory
        grid_filename: Base filename
        etopo_file: ETOPO file for land mask
        region_mask_file: Region mask file
        lat_sta: Start latitude
        lat_end: End latitude
    """
    # Write to text file
    print(f'\tWriting grid info to {grid_path}/{grid_filename}.txt ...')
    with open(f'{grid_path}{grid_filename}.txt', 'w') as file:
        file.write("# CDO description of a curvilinear grid. \n\n")
        file.write("gridtype = curvilinear\n")
        file.write("gridsize = {:.12g}\n".format(nx * ny))
        file.write("xsize = {:.6g}\n".format(ny))
        file.write("ysize = {:.6g}\n\n".format(nx))
        file.write("# xvals/yvals describe the positions of the lon/lat grid cells. "
                   "The first 4 values of xbounds/ybounds are the corners of the first grid cell.\n\n")

        # Write longitudes
        file.write("xvals =\n")
        for row in lon:
            for value in row:
                file.write("{:.3f}\n".format(value))

        # Write latitudes
        file.write("yvals =\n")
        for row in lat:
            for value in row:
                file.write("{:.3f}\n".format(value))

        # Write longitude boundaries
        file.write("xbounds =\n")
        for row in lon_bnds:
            for value in row:
                file.write("{:.3f} {:.3f} {:.3f} {:.3f}\n".format(value[0], value[1], value[2], value[3]))

        # Write latitude boundaries
        file.write("ybounds =\n")
        for row in lat_bnds:
            for value in row:
                file.write("{:.3f} {:.3f} {:.3f} {:.3f}\n".format(value[0], value[1], value[2], value[3]))

    # Write to NetCDF file
    print(f'\tWriting grid info to {grid_path}/{grid_filename}.nc ...')
    ds = xr.Dataset()

    ds['lon'] = xr.DataArray(lon, dims=('x', 'y'),
                             attrs={'long_name': 'Longitude', 'units': 'degree_east'})
    ds['lat'] = xr.DataArray(lat, dims=('x', 'y'),
                             attrs={'long_name': 'Latitude', 'units': 'degree_north'})
    ds['lon_bnds'] = xr.DataArray(lon_bnds, dims=('x', 'y', 'vertices'),
                                  attrs={'long_name': 'Longitude of the grid vertices',
                                         'description': 'Boundary coordinates (Clockwise: top left, top right, bottom right, bottom left)'})
    ds['lat_bnds'] = xr.DataArray(lat_bnds, dims=('x', 'y', 'vertices'),
                                  attrs={'long_name': 'Latitude of the grid vertices',
                                         'description': 'Boundary coordinates (Clockwise: top left, top right, bottom right, bottom left)'})
    ds['x_c'] = xr.DataArray(xv, dims=('x', 'y'),
                             attrs={'long_name': 'projection grid x centers', 'units': 'm'})
    ds['y_c'] = xr.DataArray(yv, dims=('x', 'y'),
                             attrs={'long_name': 'projection grid y centers', 'units': 'm'})
    ds['x_bnds'] = xr.DataArray(x_bnds, dims=('x', 'y', 'vertices'),
                                attrs={'long_name': 'projection grid x vertices'})
    ds['y_bnds'] = xr.DataArray(y_bnds, dims=('x', 'y', 'vertices'),
                                attrs={'long_name': 'projection grid y vertices'})
    ds['cell_area'] = xr.DataArray(area, dims=('x', 'y'),
                                   attrs={'long_name': 'grid cell area', 'units': 'm ** 2'})

    # Add land mask if ETOPO file is provided
    if etopo_file is not None:
        print('\tComputing land mask ...')
        _add_land_mask(ds, etopo_file, grid_path, grid_filename, lat_sta, lat_end)

    # Add region mask if region mask file is provided
    if region_mask_file is not None:
        print('\tComputing region mask ...')
        _add_region_mask(ds, region_mask_file, grid_path, grid_filename)

    write_netcdf_compressed(ds, f'{grid_path}{grid_filename}.nc')

def _add_land_mask(ds: xr.Dataset, etopo_file: str, grid_path: str,
                   grid_filename: str, lat_sta: float, lat_end: float) -> None:
    """Add land mask to grid dataset.

    Args:
        ds: Grid dataset
        etopo_file: ETOPO topography file
        grid_path: Grid file directory
        grid_filename: Grid filename
        lat_sta: Start latitude
        lat_end: End latitude
    """
    # Generate a land/sea mask from ETOPO topography data:
    #   1. Clip to the target latitude range
    #   2. Set negative values (ocean) to 1 and positive values (land) to 0
    #   3. Bilinearly interpolate to the evaluation grid
    #   4. Classify interpolated values < 0.99 as land (0) and ≥ 0.99 as ocean (1)
    with tempfile.TemporaryDirectory(prefix='sitool_lm_', dir=_runtime_tmpdir()) as _tmp:
        t1 = os.path.join(_tmp, 'temp1.nc')
        t2 = os.path.join(_tmp, 'temp2.nc')
        t3 = os.path.join(_tmp, 'temp3.nc')
        t4 = os.path.join(_tmp, 'temp4.nc')
        cdo.sellonlatbox(-180, 180, lat_sta, lat_end, input=etopo_file, output=t1)
        cdo.setrtoc2(-1e6, 0, 1, 0, input=t1, output=t2)
        cdo.remapbil(f'{grid_path}{grid_filename}.txt', input=t2, output=t3)
        cdo.setrtoc2(0, 0.99, 0, 1, input=t3, output=t4)
        with xr.open_dataset(t4) as ds2:
            z = np.array(ds2['z'])
    ds['land_sea_mask'] = xr.DataArray(z, dims=('x', 'y'),
                                       attrs={'long_name': 'ocean/land mask',
                                              'description': '1: ocean grid, 0: land grid'})

def _add_region_mask(ds: xr.Dataset, region_mask_file: str,
                     grid_path: str, grid_filename: str) -> None:
    """Add region mask to grid dataset.

    Args:
        ds: Grid dataset
        region_mask_file: Region mask file
        grid_path: Grid file directory
        grid_filename: Grid filename
    """
    # Use nearest-neighbour interpolation to map the region mask onto the evaluation grid
    # (preserves discrete category values without smoothing)
    with tempfile.TemporaryDirectory(prefix='sitool_rm_', dir=_runtime_tmpdir()) as _tmp:
        t2 = os.path.join(_tmp, 'temp2.nc')
        cdo.remapnn(f'{grid_path}{grid_filename}.txt', input=region_mask_file, output=t2)
        with xr.open_dataset(t2) as ds2:
            try:
                sea_ice_region = np.array(ds2['sea_ice_region'])
            except KeyError:
                # Older NSIDC files use a slightly different variable name
                sea_ice_region = np.array(ds2['sea_ice_region_NASA'])
    ds['sea_ice_region'] = xr.DataArray(sea_ice_region, dims=('x', 'y'),
                                        attrs={'long_name': 'sea ice region mask'})

def gen_mask(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """Generate a valid-data mask for two co-located arrays.

    Cells are masked out (set to 0) when:
    - both arrays are exactly zero (ice-free ocean / open water), or
    - either array contains NaN (missing data).
    All other cells are set to 1.

    Parameters
    ----------
    array1, array2 : ndarray
        Input arrays of identical shape (1-D or 2-D).

    Returns
    -------
    mask : ndarray
        Float array of the same shape; 1 = valid, 0 = masked.
    """
    if array2.shape != array1.shape:
        raise ValueError('Input arrays must have the same shape!')

    # Vectorised: no Python-level loops needed
    both_zero  = (array1 == 0) & (array2 == 0)
    either_nan = np.isnan(array1) | np.isnan(array2)
    return np.where(both_zero | either_nan, 0.0, 1.0)

__all__ = [
    "gen_polar_grid",
    "_calculate_cell_areas",
    "_write_grid_files",
    "_add_land_mask",
    "_add_region_mask",
    "gen_mask",
]
