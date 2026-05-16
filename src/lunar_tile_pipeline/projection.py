"""Spherical lunar equirectangular projection utilities.

This first version intentionally uses a simple spherical Moon approximation.
It is deterministic and adequate for global tile indexing, but it is not an
Earth CRS and should not be treated as EPSG:4326.
"""

from __future__ import annotations

import math

from lunar_tile_pipeline.config import MOON_RADIUS_M


def normalize_lon_180(lon: float) -> float:
    """Normalize longitude to [-180, 180)."""
    return ((float(lon) + 180.0) % 360.0) - 180.0


def normalize_lon_360(lon: float) -> float:
    """Normalize longitude to [0, 360)."""
    return float(lon) % 360.0


def latlon_to_xy(lat: float, lon: float, radius_m: float = MOON_RADIUS_M) -> tuple[float, float]:
    """Project latitude/longitude to spherical lunar equirectangular meters."""
    lon_180 = normalize_lon_180(lon)
    x = radius_m * math.radians(lon_180)
    y = radius_m * math.radians(float(lat))
    return x, y


def xy_to_latlon(x: float, y: float, radius_m: float = MOON_RADIUS_M) -> tuple[float, float]:
    """Invert spherical lunar equirectangular meters to latitude/lon_180."""
    lon_180 = normalize_lon_180(math.degrees(float(x) / radius_m))
    lat = math.degrees(float(y) / radius_m)
    return lat, lon_180
