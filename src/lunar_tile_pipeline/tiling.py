"""Deterministic global lunar tile grid."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterator

from shapely.geometry import Polygon

from lunar_tile_pipeline.config import DEFAULT_TILE_SIZE_KM, MOON_RADIUS_M
from lunar_tile_pipeline.projection import latlon_to_xy, normalize_lon_180, normalize_lon_360, xy_to_latlon


@dataclass(frozen=True)
class LunarTile:
    tile_id: str
    tile_size_km: float
    tile_size_m: float
    tile_x: int
    tile_y: int
    center_x: float
    center_y: float
    center_lat: float
    center_lon_180: float
    center_lon_360: float
    bounds_xy: tuple[float, float, float, float]
    bounds_latlon: dict[str, object]
    polygon_xy: Polygon
    polygon_latlon: Polygon

    def to_metadata(self) -> dict[str, object]:
        x_min, y_min, x_max, y_max = self.bounds_xy
        return {
            "tile_id": self.tile_id,
            "tile_size_km": self.tile_size_km,
            "tile_x": self.tile_x,
            "tile_y": self.tile_y,
            "projection": "spherical_equirectangular_moon_first_version",
            "moon_radius_m": MOON_RADIUS_M,
            "center": {
                "x_m": self.center_x,
                "y_m": self.center_y,
                "lat": self.center_lat,
                "lon_180": self.center_lon_180,
                "lon_360": self.center_lon_360,
            },
            "bounds_xy_m": {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
            },
            "bounds_latlon": self.bounds_latlon,
        }


def tile_size_to_m(tile_size_km: float) -> float:
    return float(tile_size_km) * 1000.0


def format_tile_size(tile_size_km: float) -> str:
    value = float(tile_size_km)
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def make_tile_id(tile_x: int, tile_y: int, tile_size_km: float) -> str:
    return f"moon_{format_tile_size(tile_size_km)}km_x{tile_x:+07d}_y{tile_y:+07d}"


def get_tile_for_latlon(lat: float, lon: float, tile_size_km: float = DEFAULT_TILE_SIZE_KM) -> LunarTile:
    x, y = latlon_to_xy(lat, lon)
    return get_tile_for_xy(x, y, tile_size_km=tile_size_km)


def get_tile_for_xy(x: float, y: float, tile_size_km: float = DEFAULT_TILE_SIZE_KM) -> LunarTile:
    tile_size_m = tile_size_to_m(tile_size_km)
    tile_x = math.floor((float(x) + tile_size_m / 2.0) / tile_size_m)
    tile_y = math.floor((float(y) + tile_size_m / 2.0) / tile_size_m)
    return get_tile_by_indices(tile_x, tile_y, tile_size_km=tile_size_km)


def get_tile_by_indices(tile_x: int, tile_y: int, tile_size_km: float = DEFAULT_TILE_SIZE_KM) -> LunarTile:
    tile_size_m = tile_size_to_m(tile_size_km)
    center_x = int(tile_x) * tile_size_m
    center_y = int(tile_y) * tile_size_m
    half = tile_size_m / 2.0
    x_min = center_x - half
    x_max = center_x + half
    y_min = center_y - half
    y_max = center_y + half
    center_lat, center_lon_180 = xy_to_latlon(center_x, center_y)
    center_lon_360 = normalize_lon_360(center_lon_180)

    xy_corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)]
    latlon_corners = [_xy_corner_to_lonlat(corner_x, corner_y) for corner_x, corner_y in xy_corners]
    corner_latlon = [{"lat": lat, "lon_180": lon_180, "lon_360": normalize_lon_360(lon_180)} for lon_180, lat in latlon_corners]
    lats = [item["lat"] for item in corner_latlon]
    lon180s = [item["lon_180"] for item in corner_latlon]
    lon360s = [item["lon_360"] for item in corner_latlon]
    crosses_lon_360_wrap = (max(lon360s) - min(lon360s)) > 180.0

    polygon_xy = Polygon(xy_corners)
    polygon_latlon = Polygon(latlon_corners)
    bounds_latlon = {
        "corner_latlon": corner_latlon,
        "min_lat_approx": min(lats),
        "max_lat_approx": max(lats),
        "min_lon_180_approx": min(lon180s),
        "max_lon_180_approx": max(lon180s),
        "min_lon_360_approx": max(lon360s) if crosses_lon_360_wrap else min(lon360s),
        "max_lon_360_approx": min(lon360s) if crosses_lon_360_wrap else max(lon360s),
        "crosses_lon_360_wrap": crosses_lon_360_wrap,
    }

    return LunarTile(
        tile_id=make_tile_id(tile_x, tile_y, tile_size_km),
        tile_size_km=float(tile_size_km),
        tile_size_m=tile_size_m,
        tile_x=int(tile_x),
        tile_y=int(tile_y),
        center_x=center_x,
        center_y=center_y,
        center_lat=center_lat,
        center_lon_180=center_lon_180,
        center_lon_360=center_lon_360,
        bounds_xy=(x_min, y_min, x_max, y_max),
        bounds_latlon=bounds_latlon,
        polygon_xy=polygon_xy,
        polygon_latlon=polygon_latlon,
    )


def _xy_corner_to_lonlat(x: float, y: float) -> tuple[float, float]:
    lat, lon_180 = xy_to_latlon(x, y)
    return lon_180, lat


def iter_tiles_for_latlon_bounds(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    tile_size_km: float = DEFAULT_TILE_SIZE_KM,
) -> Iterator[LunarTile]:
    lon_candidates = [normalize_lon_180(min_lon), normalize_lon_180(max_lon)]
    if lon_candidates[1] < lon_candidates[0]:
        lon_candidates = [lon_candidates[0], 180.0, -180.0, lon_candidates[1]]
    xs = [latlon_to_xy(lat, lon)[0] for lat in (min_lat, max_lat) for lon in lon_candidates]
    ys = [latlon_to_xy(lat, lon)[1] for lat in (min_lat, max_lat) for lon in lon_candidates]
    min_tile = get_tile_for_xy(min(xs), min(ys), tile_size_km=tile_size_km)
    max_tile = get_tile_for_xy(max(xs), max(ys), tile_size_km=tile_size_km)
    min_lat_norm = min(min_lat, max_lat)
    max_lat_norm = max(min_lat, max_lat)

    for tile_y in range(min_tile.tile_y, max_tile.tile_y + 1):
        for tile_x in range(min_tile.tile_x, max_tile.tile_x + 1):
            tile = get_tile_by_indices(tile_x, tile_y, tile_size_km=tile_size_km)
            if min_lat_norm <= tile.center_lat <= max_lat_norm:
                yield tile


def iter_global_tiles(tile_size_km: float = DEFAULT_TILE_SIZE_KM) -> Iterator[LunarTile]:
    x_min = -math.pi * MOON_RADIUS_M
    x_max = math.pi * MOON_RADIUS_M
    y_min = -math.pi * MOON_RADIUS_M / 2.0
    y_max = math.pi * MOON_RADIUS_M / 2.0
    min_tile = get_tile_for_xy(x_min, y_min, tile_size_km=tile_size_km)
    max_tile = get_tile_for_xy(x_max, y_max, tile_size_km=tile_size_km)
    for tile_y in range(min_tile.tile_y, max_tile.tile_y + 1):
        for tile_x in range(min_tile.tile_x, max_tile.tile_x + 1):
            tile = get_tile_by_indices(tile_x, tile_y, tile_size_km=tile_size_km)
            if -90.0 <= tile.center_lat <= 90.0 and -180.0 <= tile.center_lon_180 < 180.0:
                yield tile


def parse_tile_id(tile_id: str) -> tuple[float, int, int]:
    match = re.fullmatch(r"moon_(?P<size>\d+(?:p\d+)?)km_x(?P<x>[+-]\d+)_y(?P<y>[+-]\d+)", tile_id)
    if not match:
        raise ValueError(f"Invalid tile_id: {tile_id}")
    size_text = match.group("size").replace("p", ".")
    return float(size_text), int(match.group("x")), int(match.group("y"))
