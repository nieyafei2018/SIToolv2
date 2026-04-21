# -*- coding: utf-8 -*-
"""Split utility helpers."""

import os
from typing import Dict, List, Tuple

from scripts.utils import core as _core
from scripts.utils.grid import gen_mask

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


def MatrixDiff(array1: np.ndarray, array2: np.ndarray, weights: np.ndarray = None,
               metric: str = 'MAE', mask: bool = False):
    """
    Purpose
    -------
    Calculate the weighted difference between two arrays (1-d or 2-d).

    Returns
    -------
    The weighted statistical metric for measuring the difference/correlation between A1 and A2

    Example
    -------
    >> A1 = np.array([[1, 0, 4, 5, np.nan], [2, np.nan, 8, 6, np.nan]])
    >> A2 = np.array([[2, 3, 0, 9, 10.], [np.nan, 1, -1, 4, 2]])
    >> weight = np.array([[1., 1, 1, 2, 1], [2, 3, 1, 1, 3]])

    >> value = SITool_basic().Diff_2Matrix(A1, A2, weight, stats='RMSE')
    >> print(value)

    the output will be
    4.5197977
    """
    use_mask = bool(mask)

    # Build a spatial mask: exclude cells where both arrays are zero (ice-free)
    # or where either array contains NaN
    if use_mask:
        mask = gen_mask(array1, array2)
    else:
        mask = np.ones(array1.shape)

    if weights is None:
        weights = np.ones(array1.shape)

    # Flatten to 1-D for vectorised operations
    array1, array2 = np.array(array1).reshape(-1, ), np.array(array2).reshape(-1, )
    weights, mask = np.array(weights).reshape(-1, ), np.array(mask).reshape(-1, )

    if (array1.shape != array2.shape) or (array1.shape != weights.shape):
        raise ValueError('Error: Input arrays and weights must have the same shape!')

    if (not use_mask) and (np.any(array1 == 0) or np.any(array2 == 0)):
        logger.warning("Input arrays contain 0 values without masking — verify this does not affect results.")

    # Remove NaN pairs before computing the metric
    index = ~(np.isnan(array1) | np.isnan(array2))
    x, y, weights, mask = array1[index], array2[index], weights[index], mask[index]

    if metric == 'Bias':
        # Weighted mean bias (signed)
        return np.sum((y - x) * weights * mask) / np.sum(weights * mask)
    elif metric == 'MAE':
        # Weighted mean absolute error
        return np.sum(abs(x - y) * weights * mask) / np.sum(weights * mask)
    elif metric == 'RMSE':
        # Weighted root-mean-square error
        return np.sqrt(np.sum((x - y) ** 2 * weights * mask) / np.sum(weights * mask))
    elif metric == 'Corr':
        # Weighted Pearson correlation
        weighted_corr, p_value = pearsonr(x * weights, y * weights)
        return {'weighted_corr': weighted_corr, 'p_value': p_value}
    else:
        raise ValueError(f'Error: The statistical metric must be one of the "Bias", "MAE", "RMSE", and "Corr"!')

def median_filter(A, n):
    """
    Purpose
    -------
    Apply n x n grid filter on 2-d matrix A

    Parameters
    ----------
    A: 2-d array
    n: int value
       must be an odd number

    Returns
    -------
    C: 2-d array

    Notes
    -----
    The processing of boundary points please refer to the following Example.

    Example
    -------
    >> rows = 3
    >> cols = 4
    >> A = [[np.random.randint(0, 9) for _ in range(cols)] for _ in range(rows)]
    >> As = SITool_basic().median_filter(A, 3)
    >> print('Original matrix:', A)
    >> print('Smoothed matrix:', As)

    The Output could be:
    Original matrix: [[5, 0, 3, 3],
                      [7, 3, 5, 2],
                      [4, 7, 6, 8]]
    Smoothed matrix: [[3.75       3.83333333 2.66666667 3.25      ]
                     [4.33333333 4.44444444 4.11111111 4.5       ]
                     [5.25       5.33333333 5.16666667 5.25      ]]
    in which the value 3.75 is an average of [[5, 0], [7, 3]], and 3.83333 is an average of [[5, 0, 3], [7, 3, 5]]
    """

    nx, ny = np.shape(A)
    k = int((n - 1) / 2)  # filter radius

    # Pad with NaN so the filter window shrinks automatically at the edges
    # (nanmean ignores NaN, giving a smaller effective window near boundaries)
    B = np.full((nx + k * 2, ny + k * 2), np.nan)
    B[k:k + nx, k:k + ny] = A

    # Stack n×n shifted views along a third axis, then take nanmean to apply the mean filter
    C = np.nanmean(np.stack([B[i:i + nx, j:j + ny] for i in range(n) for j in range(n)], axis=2), axis=2)

    return C

def ll_dist_matrix(lon, lat):
    """Compute cumulative distance matrices in the x and y directions for a 2-D grid (Haversine formula).

    These are the key geometric quantities for sea-ice advection/divergence calculations:
      hx[m, n] = cumulative great-circle distance from row 0 to row m at column n (x-direction)
      hy[m, n] = cumulative great-circle distance from column 0 to column n at row m (y-direction)

    Args:
        lon, lat: 2-D longitude/latitude arrays (nx, ny).

    Returns:
        (hx, hy): Cumulative distance matrices, units [m] (internally multiplied by 1000 to convert km → m).
    """
    nx, ny = np.shape(lon)

    dx = np.full((nx, ny), np.nan)
    dy = np.full((nx, ny), np.nan)
    hx = np.full((nx, ny), np.nan)
    hy = np.full((nx, ny), np.nan)

    def ll_distance_hav(lat0, lng0, lat1, lng1):
        """
        Calculate distance between (lng0, lat0) and (lng1, lat1)

        Method: haversine formulation
        unit [km]
        ref:
            https://blog.csdn.net/qq_35462323/article/details/106525763
        """

        EARTH_RADIUS = 6378.137
        lat0 = radians(lat0)
        lat1 = radians(lat1)
        lng0 = radians(lng0)
        lng1 = radians(lng1)
        dlng = fabs(lng0 - lng1)
        dlat = fabs(lat0 - lat1)
        h = sin(dlat / 2) * sin(dlat / 2) + cos(lat0) * cos(lat1) * sin(dlng / 2) * sin(dlng / 2)

        return 2 * EARTH_RADIUS * asin(sqrt(h))

    # hx[m, n] = cumulative distance from row 0 to row m at column n (x-direction)
    for m in range(1, nx):
        for n in range(0, ny):
            dx[m, n] = ll_distance_hav(lat[m, n], lon[m, n],
                                       lat[m - 1, n], lon[m - 1, n]) * 1000
        hx[m, :] = np.nansum(dx[0:m + 1, :], axis=0)

    # hy[m, n] = cumulative distance from column 0 to column n at row m (y-direction)
    for n in range(1, ny):
        for m in range(0, nx):
            dy[m, n] = ll_distance_hav(lat[m, n], lon[m, n],
                                       lat[m, n - 1], lon[m, n - 1]) * 1000
        hy[:, n] = np.nansum(dy[:, 0:n + 1], axis=1)

    return hx, hy

def xy_gradient(A, h, direc):
    """Compute the spatial gradient of 2-D field A along the specified direction (central differencing).

    Formula: dA/dh ≈ (A[i+1] - A[i-1]) / (h[i+1] - h[i-1])
    Boundary rows/columns cannot be centred and remain NaN.

    Args:
        A:     2-D field to differentiate (nx, ny).
        h:     Cumulative distance matrix (nx, ny) in the corresponding direction,
               as produced by ll_dist_matrix.
        direc: 'x' to differentiate along rows; 'y' to differentiate along columns.

    Returns:
        dA: Gradient field (nx, ny), units = [A units] / [h units].
    """
    nx, ny = np.shape(A)
    dA = np.full((nx, ny), np.nan)

    # Central differencing: dA/dh ≈ (A[i+1] - A[i-1]) / (h[i+1] - h[i-1])
    # Boundary rows/columns remain NaN (no valid neighbours)
    if direc == 'x':
        dA[1:nx - 1, :] = (A[2:nx, :] - A[0:nx - 2, :]) / (h[2:nx, :] - h[0:nx - 2, :])
    elif direc == 'y':
        dA[:, 1:ny - 1] = (A[:, 2:ny] - A[:, 0:ny - 2]) / (h[:, 2:ny] - h[:, 0:ny - 2])

    return dA

_SECTOR_DEFINITIONS: Dict[str, List[Tuple[str, str]]] = {
    # Southern Hemisphere sectors used by SITool diagnostics.
    'sh': [
        ('All', 'All Regions'),
        ('Weddell', 'Weddell Sea'),
        ('West Indian', 'West Indian Ocean'),
        ('East Indian', 'East Indian Ocean'),
        ('Ross', 'Ross Sea'),
        ('AB', 'Amundsen and Bellingshausen Seas'),
    ],
    # Northern Hemisphere sectors used by SITool diagnostics.
    # These follow the grouped sectors currently used by region_index().
    'nh': [
        ('All', 'All Regions'),
        ('CA', 'Central Arctic'),
        ('CBS', 'Chukchi/Beaufort Seas'),
        ('LESS', 'Laptev/East Siberian Seas'),
        ('KBS', 'Kara/Barents Seas'),
        ('EG', 'East Greenland Sea'),
        ('HBB', 'Hudson/Baffin Bays'),
        ('CAA', 'Canadian Arctic Archipelago'),
        ('Other', 'Other NSIDC Regions'),
    ],
}


def get_hemisphere_sectors(hms: str, include_all: bool = True) -> List[str]:
    """Return supported sector keys for one hemisphere.

    Args:
        hms: Hemisphere key ('nh' or 'sh').
        include_all: Whether to keep the domain-wide 'All' entry.
    """
    h = str(hms or '').lower()
    defs = _SECTOR_DEFINITIONS.get(h)
    if defs is None:
        raise ValueError(f'Unsupported hemisphere: {hms!r}. Expected "nh" or "sh".')
    sectors = [key for key, _ in defs]
    if include_all:
        return sectors
    return [s for s in sectors if s != 'All']


def get_sector_label(hms: str, sector: str) -> str:
    """Return human-readable label for a sector key."""
    h = str(hms or '').lower()
    s = str(sector or '').strip()
    defs = _SECTOR_DEFINITIONS.get(h, [])
    for key, label in defs:
        if key == s:
            return label
    return s


def region_index(grid_file, hms, sector='All'):
    """
    Purpose
    -------
    Calculate the index of the different geographical sea sectors defined by the NSIDC.

    Parameters
    ----------
    grid_file: <str>
    hms: <str>
    sector: <str>

    Return
    ------
    index: <2-d array>, in the same shape of lon and lat,
        If a grid is in the specific sector, the value is True, else False.

    References
    ----------
    Meier, W. N. and J. S. Stewart. (2023). Arctic and Antarctic Regional Masks for Sea Ice and Related
        Data Products, Version 1 [Data Set]. Boulder, Colorado USA. National Snow and Ice Data Center.
        https://doi.org/10.5067/CYW3O8ZUNIWC.
    """

    with xr.open_dataset(grid_file) as ds:
        sea_ice_region = np.array(ds['sea_ice_region'])

    hms = str(hms or '').lower()
    sector = str(sector or 'All').strip()

    if hms == 'sh':
        '''
        Weddell Sea:                             (lon > 300) | (lon <= 20)
        West Indian:                             20 < lon <= 90
        East Indian:                             90 < lon <= 160
        Ross Sea:                                160 < lon <= 230
        Amundsen_and Bellingshausen Seas (AB):   230 < lon <= 300
        '''
        mappings = {
            'All': sea_ice_region > 0,
            'Weddell': sea_ice_region == 1,
            'West Indian': sea_ice_region == 2,
            'East Indian': sea_ice_region == 3,
            'Ross': sea_ice_region == 4,
            'AB': sea_ice_region == 5,
        }
        if sector not in mappings:
            raise ValueError(
                f'Unsupported sector "{sector}" for hemisphere "{hms}". '
                f'Available: {", ".join(get_hemisphere_sectors(hms))}'
            )
        index = mappings[sector]

    elif hms == 'nh':
        '''
        Central Arctic (CA):
        Chukchi/Beaufort Seas (CBS):         305 < lon <= 360, lat < 74
        Laptev/East Siberian Seas (LESS):    0 < lon <= 75, lat < 74 / 78
        Kara/Barents Seas (KBS):             75 < lon <= 160, lat < 78
        East Greenland (EG):                 160 < lon <= 225, lat < 78
        Hudson/Baffin Bays (HBB):            225 < lon <= 280, 50 < lat < 70 / 78
        Canadian Arctic Archipelago (CAA):   260 < lon <= 310, 64 < lat < 78
        Other NSIDC regions (Other):         sea_ice_region in [13, 14, 15, 17, 18]
        '''
        mappings = {
            'All': sea_ice_region > 0,
            'CA': sea_ice_region == 1,
            'CBS': (sea_ice_region == 2) | (sea_ice_region == 3),
            'LESS': (sea_ice_region == 4) | (sea_ice_region == 5),
            'KBS': (sea_ice_region == 6) | (sea_ice_region == 7),
            'EG': sea_ice_region == 8,
            'HBB': (sea_ice_region == 9) | (sea_ice_region == 11),
            'CAA': sea_ice_region == 12,
            'Other': np.isin(sea_ice_region, [13, 14, 15, 17, 18]),
        }
        if sector not in mappings:
            raise ValueError(
                f'Unsupported sector "{sector}" for hemisphere "{hms}". '
                f'Available: {", ".join(get_hemisphere_sectors(hms))}'
            )
        index = mappings[sector]
    else:
        raise ValueError(f'Unsupported hemisphere: {hms!r}. Expected "nh" or "sh".')

    return np.asarray(index, dtype=bool)

def adaptive_row_col(N: int) -> tuple:
    """Compute a compact (rows, cols) layout for N subplots.

    Tries to keep the grid as square as possible, capping columns at 5
    so figures do not become too wide on standard screens.

    Parameters
    ----------
    N : int
        Number of subplots to arrange.

    Returns
    -------
    rows, cols : int
        Grid dimensions satisfying rows * cols >= N.
    """
    rows = math.ceil(math.sqrt(N))
    cols = math.ceil(N / rows)
    if cols > 5:          # cap at 5 columns to avoid overly wide figures
        cols = 5
        rows = math.ceil(N / cols)
    return rows, cols

def extract_grid(nc_file, lon_name, lat_name, grid_file):
    """

    :return:
    """
    # --- Open file and read variables ---
    try:
        with xr.open_dataset(nc_file) as ds:
            lon, lat = np.array(ds[lon_name]), np.array(ds[lat_name])
    except Exception:
        # Some files use non-standard time encoding; decode_times=False bypasses that
        with xr.open_dataset(nc_file, decode_times=False) as ds:
            lon, lat = np.array(ds[lon_name]), np.array(ds[lat_name])

    if lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    nx, ny = lon.shape

    # --- Write into txt ---
    print(f'\tWriting the grid info into {grid_file}.txt ...')
    with open(f'{grid_file}.txt', 'w') as file:
        file.write("# CDO description of a curvilinear grid. \n\n")
        file.write("gridtype = curvilinear\n")
        file.write("gridsize = {:.12g}\n".format(nx * ny))
        file.write("xsize = {:.6g}\n".format(ny))
        file.write("ysize = {:.6g}\n\n".format(nx))
        file.write("# xvals/yvals describe the positions of the lon/lat grid cells. "
                   "The first 4 values of xbounds/ybounds are the corners of the first grid cell.\n\n")

        # write longitudes
        file.write("xvals =\n")
        for row in lon:
            for value in row:
                file.write("{:.3f}\n".format(value))

        # write latitudes
        file.write("yvals =\n")
        for row in lat:
            for value in row:
                file.write("{:.3f}\n".format(value))

    return lon, lat

def cal_ss_clim(data, season_dict, threshold=None):
    """Calculate seasonal climatology means.

    Args:
        data: xarray DataArray with a 'time' dimension.
        season_dict: Dict mapping season name to 0-based month indices.
        threshold: If provided, mask values below this threshold before averaging.

    Returns:
        Dict mapping season name to climatology DataArray.
    """
    if threshold:
        data = data.where(data > threshold, np.nan)

    seasonal_clim = {}

    for season_name, month_list in season_dict.items():
        new_list = [x + 1 for x in month_list]

        # Create season mask
        mask = data['time.month'].isin(new_list)

        if not (~mask).all():
            # Select all data for this season and compute mean
            seasonal_data = data.where(mask, drop=True)
            seasonal_clim[season_name] = seasonal_data.mean(dim='time', skipna=True)
        else:
            # No data for this season — fill with NaN
            expected_dims = [dim for dim in data.dims if dim != 'time']
            expected_shape = tuple(data.sizes[dim] for dim in expected_dims)
            nan_array = np.full(expected_shape, np.nan, dtype=data.dtype)
            expected_coords = {dim: data.coords[dim] for dim in expected_dims}
            seasonal_clim[season_name] = xr.DataArray(
                nan_array,
                dims=expected_dims,
                coords=expected_coords,
                attrs={'description': f'Season {season_name} has no data'},
            )

        seasonal_clim[season_name].attrs['season'] = season_name
        seasonal_clim[season_name].attrs['months'] = str(new_list)

    return seasonal_clim

def cal_ss_clim_cdo(data_file, var, ss_num, threshold=None):
    """Calculate seasonal climatology means using CDO.

    Args:
        data_file: Input NetCDF file path.
        var: Variable name to extract.
        ss_num: Dict mapping season name to 0-based month indices.
        threshold: If provided, mask values below this threshold.

    Returns:
        Array of shape (n_seasons, nx, ny) with seasonal climatologies.
    """
    # Use a unique temp file to avoid collisions
    fd, tmp_path = tempfile.mkstemp(suffix='.nc', prefix='sitool_clim_', dir=_runtime_tmpdir())
    os.close(fd)
    try:
        cdo.ymonmean(input=data_file, output=tmp_path)

        with xr.open_dataset(tmp_path) as ds:
            if threshold:
                ds = ds.where(ds > threshold, np.nan)

            data1 = np.array(ds[var])
            seasons = list(ss_num.keys())

            ns = len(ss_num)
            _, nx, ny = data1.shape

            data2 = np.full((ns, nx, ny), np.nan)
            for ss in range(ns):
                logger.debug("\t%s %s", seasons[ss], np.array(ss_num[seasons[ss]]) + 1)
                data2[ss, :, :] = np.mean(data1[ss_num[seasons[ss]], :, :], axis=0)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return data2

def stable_interpolation(grid_file: str, input_file: str, output_file: str, cdo_instance: Cdo,
                         method: str = 'bilinear',
                         allow_fallback: bool = True) -> None:
    """Perform robust spatial interpolation with error handling.

    Args:
        grid_file: Target grid file for interpolation
        input_file: Input file to be interpolated
        output_file: Output file path
        cdo_instance: CDO instance for operations

    Raises:
        ValueError: If interpolation fails after all attempts
    """
    method_norm = str(method or 'bilinear').strip().lower()
    if method_norm in {'bilinear', 'bil', 'remapbil'}:
        method_norm = 'bilinear'
    elif method_norm in {'conservative', 'con', 'remapcon'}:
        method_norm = 'conservative'
    else:
        raise ValueError(
            "stable_interpolation(method=...) must be one of "
            "'bilinear' or 'conservative'."
        )

    current_input = input_file
    _tmp_delvar = None  # process-unique temp file created only if needed

    try:
        while True:
            try:
                # Use absolute paths to ensure CDO can find files
                if method_norm == 'conservative':
                    cdo_instance.remapcon(grid_file, input=current_input, output=output_file)
                else:
                    cdo_instance.remapbil(grid_file, input=current_input, output=output_file)
                break
            except Exception as e:
                error_msg = str(e)

                # Handle non-interpolatable variables
                if "Unsupported generic coordinates" in error_msg:
                    match = re.search(r"Variable:\s*([^)\s!]+)", error_msg)
                    if match:
                        problematic_var = match.group(1).strip()
                        logger.warning("Removing unsupported variable: %s", problematic_var)
                        # Use a guaranteed-unique temp file to avoid cross-process collisions
                        fd, _tmp_delvar = tempfile.mkstemp(
                            suffix='.nc', prefix='sitool_delvar_', dir=_runtime_tmpdir()
                        )
                        os.close(fd)
                        cdo_instance.delvar(problematic_var, input=current_input, output=_tmp_delvar)
                        current_input = _tmp_delvar
                        continue

                if method_norm == 'bilinear' and "unstructured source grids" in error_msg:
                    logger.info("Trying conservative remapping ...")
                    try:
                        cdo_instance.remapcon(grid_file, input=current_input, output=output_file)
                        break
                    except Exception:
                        raise ValueError(f"Failed to interpolate {input_file}")

                if method_norm == 'conservative' and allow_fallback:
                    logger.warning(
                        "Conservative remapping failed for %s (%s). "
                        "Falling back to bilinear remapping.",
                        input_file, error_msg,
                    )
                    method_norm = 'bilinear'
                    continue

                raise
    finally:
        if _tmp_delvar and os.path.exists(_tmp_delvar):
            os.remove(_tmp_delvar)

__all__ = [
    "MatrixDiff",
    "median_filter",
    "ll_dist_matrix",
    "xy_gradient",
    "get_hemisphere_sectors",
    "get_sector_label",
    "region_index",
    "adaptive_row_col",
    "extract_grid",
    "cal_ss_clim",
    "cal_ss_clim_cdo",
    "stable_interpolation",
]
