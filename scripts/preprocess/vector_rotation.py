# -*- coding: utf-8 -*-
"""Vector-frame inference and grid-aware rotation helpers for SIdrift preprocessing."""

from typing import Dict, Optional, Tuple

import numpy as np


def infer_source_frame_from_attrs(u_attrs: Dict, v_attrs: Dict) -> Tuple[Optional[str], str]:
    """Infer vector source frame from CF/metadata text.

    Returns
    -------
    (frame, reason)
      frame: one of {'xy', 'lonlat', None}
    """

    def _merged_text(attrs: Dict) -> str:
        if not attrs:
            return ""
        parts = []
        for key, value in attrs.items():
            parts.append(str(key).lower())
            parts.append(str(value).lower())
        return " ".join(parts)

    u_txt = _merged_text(u_attrs)
    v_txt = _merged_text(v_attrs)
    merged = f"{u_txt} {v_txt}"

    if (
        ("eastward" in u_txt or "zonal" in u_txt)
        and ("northward" in v_txt or "meridional" in v_txt)
    ):
        return "lonlat", "eastward/northward metadata"

    if "sea_ice_x_velocity" in u_txt and "sea_ice_y_velocity" in v_txt:
        return "xy", "sea_ice_x/y_velocity metadata"

    if (
        ("x-component" in u_txt and "y-component" in v_txt)
        or ("native model grid" in merged)
        or ("native model" in merged and "x-velocity" in merged)
    ):
        return "xy", "x/y native-grid metadata"

    return None, "no decisive metadata"


def resolve_source_frame(
    requested: Optional[str],
    inferred: Optional[str],
    *,
    prefer_inferred: bool = True,
) -> Tuple[str, str]:
    """Resolve effective source frame from recipe request and metadata inference."""
    req = str(requested or "").strip().lower()

    if req not in ("xy", "lonlat", "other", "auto", ""):
        req = "auto"

    if req in ("", "auto"):
        if inferred is not None:
            return inferred, "metadata"
        return "xy", "default_xy"

    if req == "other":
        if inferred is not None:
            return inferred, "other_fallback_to_metadata"
        return "xy", "other_fallback_to_xy"

    if inferred is not None and inferred != req and prefer_inferred:
        return inferred, "metadata_override"

    return req, "recipe"


def grid_kind_from_lonlat(lon: np.ndarray, lat: np.ndarray) -> str:
    """Classify grid as structured or unstructured based on lon/lat shapes."""
    lon_arr = np.asarray(lon)
    lat_arr = np.asarray(lat)
    if lon_arr.shape != lat_arr.shape:
        return "unknown"
    if lon_arr.ndim >= 2:
        return "structured"
    if lon_arr.ndim == 1:
        return "unstructured"
    return "unknown"


def _wrap_delta_lon_deg(dlon_deg: np.ndarray) -> np.ndarray:
    """Wrap longitude increments to [-180, 180] in degrees."""
    return (dlon_deg + 180.0) % 360.0 - 180.0


def _normalize_components(east: np.ndarray, north: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mag = np.hypot(east, north)
    east_u = np.full_like(east, np.nan, dtype=float)
    north_u = np.full_like(north, np.nan, dtype=float)
    valid = np.isfinite(mag) & (mag > 0.0)
    east_u[valid] = east[valid] / mag[valid]
    north_u[valid] = north[valid] / mag[valid]
    return east_u, north_u


def estimate_structured_basis_from_lonlat(lon: np.ndarray, lat: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Estimate local native-grid basis vectors in east/north components.

    Uses finite differences of lon/lat along x and y index directions on
    structured grids. Returned basis fields are unit vectors:
      ex = native x-axis direction in east/north
      ey = native y-axis direction in east/north
    """
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)

    if lon_arr.ndim != 2 or lat_arr.ndim != 2:
        raise ValueError("estimate_structured_basis_from_lonlat requires 2-D lon/lat arrays.")
    if lon_arr.shape != lat_arr.shape:
        raise ValueError(f"lon/lat shape mismatch: {lon_arr.shape} vs {lat_arr.shape}")

    lat_rad = np.deg2rad(lat_arr)

    # Forward/backward i-direction increments.
    dlon_fx = _wrap_delta_lon_deg(np.roll(lon_arr, -1, axis=1) - lon_arr)
    dlat_fx = np.roll(lat_arr, -1, axis=1) - lat_arr
    dlon_bx = _wrap_delta_lon_deg(lon_arr - np.roll(lon_arr, 1, axis=1))
    dlat_bx = lat_arr - np.roll(lat_arr, 1, axis=1)

    use_fwd_x = np.isfinite(dlon_fx) & np.isfinite(dlat_fx)
    dlon_x = np.where(use_fwd_x, dlon_fx, dlon_bx)
    dlat_x = np.where(use_fwd_x, dlat_fx, dlat_bx)

    # Forward/backward j-direction increments.
    dlon_fy = _wrap_delta_lon_deg(np.roll(lon_arr, -1, axis=0) - lon_arr)
    dlat_fy = np.roll(lat_arr, -1, axis=0) - lat_arr
    dlon_by = _wrap_delta_lon_deg(lon_arr - np.roll(lon_arr, 1, axis=0))
    dlat_by = lat_arr - np.roll(lat_arr, 1, axis=0)

    use_fwd_y = np.isfinite(dlon_fy) & np.isfinite(dlat_fy)
    dlon_y = np.where(use_fwd_y, dlon_fy, dlon_by)
    dlat_y = np.where(use_fwd_y, dlat_fy, dlat_by)

    # Convert lon/lat increments to local east/north components (radian distances on unit sphere).
    gx_east = np.deg2rad(dlon_x) * np.cos(lat_rad)
    gx_north = np.deg2rad(dlat_x)
    gy_east = np.deg2rad(dlon_y) * np.cos(lat_rad)
    gy_north = np.deg2rad(dlat_y)

    ex_east, ex_north = _normalize_components(gx_east, gx_north)
    ey_east, ey_north = _normalize_components(gy_east, gy_north)

    dot = ex_east * ey_east + ex_north * ey_north
    cross = ex_east * ey_north - ex_north * ey_east

    qc = {
        "mean_abs_dot": float(np.nanmean(np.abs(dot))),
        "mean_cross": float(np.nanmean(cross)),
        "valid_fraction": float(np.mean(np.isfinite(ex_east) & np.isfinite(ey_east))),
    }

    basis = {
        "ex_east": ex_east,
        "ex_north": ex_north,
        "ey_east": ey_east,
        "ey_north": ey_north,
    }
    return basis, qc


def rotate_native_xy_to_eastnorth(u: np.ndarray, v: np.ndarray, basis: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate native x/y vector components into east/north using a basis field."""
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)

    if u_arr.shape != v_arr.shape:
        raise ValueError(f"u/v shape mismatch: {u_arr.shape} vs {v_arr.shape}")

    ex_e = np.asarray(basis["ex_east"], dtype=float)
    ex_n = np.asarray(basis["ex_north"], dtype=float)
    ey_e = np.asarray(basis["ey_east"], dtype=float)
    ey_n = np.asarray(basis["ey_north"], dtype=float)

    spatial_shape = ex_e.shape
    if u_arr.shape[-len(spatial_shape):] != spatial_shape:
        raise ValueError(
            f"basis shape {spatial_shape} does not match trailing u/v shape {u_arr.shape}"
        )

    # Broadcast basis over potential leading dimensions (e.g., time).
    lead_ndim = u_arr.ndim - ex_e.ndim
    reshape = (1,) * lead_ndim + ex_e.shape
    ex_e_b = ex_e.reshape(reshape)
    ex_n_b = ex_n.reshape(reshape)
    ey_e_b = ey_e.reshape(reshape)
    ey_n_b = ey_n.reshape(reshape)

    u_east = u_arr * ex_e_b + v_arr * ey_e_b
    v_north = u_arr * ex_n_b + v_arr * ey_n_b
    return u_east, v_north


def rotate_by_angle(u: np.ndarray, v: np.ndarray, theta: np.ndarray, theta_unit: str = "auto") -> Tuple[np.ndarray, np.ndarray]:
    """Rotate native x/y vectors by angle theta into east/north.

    Applies:
      east  = u*cos(theta) - v*sin(theta)
      north = u*sin(theta) + v*cos(theta)
    """
    if theta_unit not in ("auto", "radian", "degree"):
        raise ValueError("theta_unit must be one of 'auto', 'radian', 'degree'.")

    theta_arr = np.asarray(theta, dtype=float)
    if theta_unit == "degree":
        theta_rad = np.deg2rad(theta_arr)
    elif theta_unit == "radian":
        theta_rad = theta_arr
    else:
        max_abs = float(np.nanmax(np.abs(theta_arr))) if theta_arr.size else 0.0
        theta_rad = np.deg2rad(theta_arr) if max_abs > (2 * np.pi + 0.5) else theta_arr

    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)

    # Broadcast theta onto u/v trailing dimensions.
    if theta_rad.shape != u_arr.shape:
        tail = u_arr.shape[-theta_rad.ndim:] if theta_rad.ndim > 0 else ()
        if theta_rad.shape != tail:
            raise ValueError(
                f"theta shape {theta_rad.shape} incompatible with u/v shape {u_arr.shape}."
            )
        reshape = (1,) * (u_arr.ndim - theta_rad.ndim) + theta_rad.shape
        theta_rad = theta_rad.reshape(reshape)

    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    east = u_arr * c - v_arr * s
    north = u_arr * s + v_arr * c
    return east, north
