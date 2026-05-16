"""Footprint loading and geometry matching helpers."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon

from lunar_tile_pipeline.projection import normalize_lon_180, normalize_lon_360
from lunar_tile_pipeline.tiling import LunarTile


def load_footprints(path: str | Path) -> gpd.GeoDataFrame:
    footprints = gpd.read_file(path)
    if footprints.empty:
        return footprints
    if footprints.geometry.name is None:
        raise ValueError(f"Footprint file has no geometry column: {path}")
    return footprints


def guess_longitude_domain(gdf: gpd.GeoDataFrame) -> str:
    minx, _, maxx, _ = gdf.total_bounds
    if minx >= 0.0 and maxx > 180.0:
        return "360"
    return "180"


def tile_polygon_for_footprints(tile: LunarTile, footprints_gdf: gpd.GeoDataFrame) -> Polygon:
    domain = guess_longitude_domain(footprints_gdf)
    coords = []
    for item in tile.bounds_latlon["corner_latlon"]:
        lon = normalize_lon_360(item["lon_180"]) if domain == "360" else normalize_lon_180(item["lon_180"])
        coords.append((lon, item["lat"]))
    return Polygon(coords)


def tile_center_point_for_footprints(tile: LunarTile, footprints_gdf: gpd.GeoDataFrame):
    from shapely.geometry import Point

    domain = guess_longitude_domain(footprints_gdf)
    lon = tile.center_lon_360 if domain == "360" else tile.center_lon_180
    return Point(lon, tile.center_lat)
