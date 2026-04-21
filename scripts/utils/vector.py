# -*- coding: utf-8 -*-
"""Split utility helpers."""

from scripts.utils import core as _core

# Reuse shared utility namespace and constants from core module.
globals().update({k: v for k, v in _core.__dict__.items() if k not in globals()})


def rotate_vector_by_angle(u: np.ndarray, v: np.ndarray, theta: np.ndarray,
                           theta_unit: str = 'auto'):
    """Rotate vector components by angle theta.

    Applies:
        u_rot = u*cos(theta) - v*sin(theta)
        v_rot = u*sin(theta) + v*cos(theta)

    Args:
        u: Input u component.
        v: Input v component.
        theta: Rotation angle array.
        theta_unit: 'radian', 'degree', or 'auto' (default).

    Returns:
        Tuple (u_rot, v_rot).
    """
    if theta_unit not in ('auto', 'radian', 'degree'):
        raise ValueError("theta_unit must be one of 'auto', 'radian', 'degree'.")

    theta_arr = np.asarray(theta)
    if theta_unit == 'degree':
        theta_rad = np.deg2rad(theta_arr)
    elif theta_unit == 'radian':
        theta_rad = theta_arr
    else:
        max_abs = float(np.nanmax(np.abs(theta_arr))) if theta_arr.size else 0.0
        # Heuristic: values beyond 2π are likely in degrees.
        theta_rad = np.deg2rad(theta_arr) if max_abs > (2 * np.pi + 0.5) else theta_arr

    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    u_rot = u * cos_theta - v * sin_theta
    v_rot = u * sin_theta + v * cos_theta
    return u_rot, v_rot

def rotate_vector_cartopy(src_crs: ccrs.CRS, target_crs: ccrs.CRS,
                          lon: np.ndarray, lat: np.ndarray,
                          u_src: np.ndarray, v_src: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate vector components from a source CRS frame to a target CRS frame.

    This helper wraps ``target_crs.transform_vectors`` and is intended for
    velocity components such as eastward/northward drift that need to be
    expressed in the x/y axes of a map projection.

    Args:
        src_crs: Source coordinate reference system of ``lon/lat`` and ``u_src/v_src``
            (for example ``ccrs.PlateCarree()``).
        target_crs: Target projection in which output vector components are returned
            (for example ``ccrs.Stereographic(...)``).
        lon: Longitude values in source CRS coordinates.
        lat: Latitude values in source CRS coordinates.
        u_src: Grid-eastward vector component in source CRS.
        v_src: Grid-northward vector component in source CRS.

    Returns:
        Tuple ``(u_tgt, v_tgt)`` where both arrays have the same shape as inputs
        and represent components aligned with target CRS x/y axes.

    Raises:
        ValueError: If input arrays do not share the same shape.
    """
    lon_arr = np.asarray(lon)
    lat_arr = np.asarray(lat)
    u_arr = np.asarray(u_src)
    v_arr = np.asarray(v_src)

    if lon_arr.shape != lat_arr.shape:
        raise ValueError(
            f"lon and lat must have the same shape, got {lon_arr.shape} and {lat_arr.shape}."
        )
    if u_arr.shape != lon_arr.shape or v_arr.shape != lon_arr.shape:
        raise ValueError(
            "u_src and v_src must match lon/lat shape. "
            f"Got lon/lat={lon_arr.shape}, u_src={u_arr.shape}, v_src={v_arr.shape}."
        )

    u_tgt, v_tgt = target_crs.transform_vectors(src_crs, lon_arr, lat_arr, u_arr, v_arr)
    return np.asarray(u_tgt), np.asarray(v_tgt)

def rotate_vector_formula(u: np.ndarray, v: np.ndarray, hemisphere: str, lons: np.ndarray):
    """Rotate velocity vectors from geographic (lon/lat) to projection (x/y) coordinates.

    For polar stereographic projections the rotation angle equals the longitude,
    with sign depending on hemisphere.

    Args:
        u: Zonal velocity component (geographic east).
        v: Meridional velocity component (geographic north).
        hemisphere: 'sh' (Southern) or 'nh' (Northern).
        lons: Longitude array (degrees).

    Returns:
        Tuple (u_rot, v_rot) in projection x/y coordinates.
    """
    # For polar stereographic: rotation angle = longitude (SH: clockwise → negative)
    cos_angle = np.cos(np.radians(lons))
    sin_angle = np.sin(np.radians(lons))

    # Rotate geographic components (east/north) into projection x/y.
    if hemisphere == 'sh':
        u_rot = u * cos_angle + v * sin_angle
        v_rot = - u * sin_angle + v * cos_angle 
    elif hemisphere == 'nh':
        u_rot = u * cos_angle - v * sin_angle
        v_rot = u * sin_angle + v * cos_angle

    return u_rot, v_rot

__all__ = [
    "rotate_vector_by_angle",
    "rotate_vector_cartopy",
    "rotate_vector_formula",
]
