"""Mask and label coordinate conversion utilities."""

from __future__ import annotations

import numpy as np

from lunar_tile_pipeline.projection import normalize_lon_360, xy_to_latlon
from lunar_tile_pipeline.tiling import LunarTile


def pixel_to_tile_xy(row: float, col: float, tile: LunarTile, width_px: int, height_px: int) -> tuple[float, float]:
    x_min, y_min, x_max, y_max = tile.bounds_xy
    x = x_min + (float(col) + 0.5) * (x_max - x_min) / float(width_px)
    y = y_max - (float(row) + 0.5) * (y_max - y_min) / float(height_px)
    return x, y


def tile_xy_to_pixel(x: float, y: float, tile: LunarTile, width_px: int, height_px: int) -> tuple[int, int]:
    x_min, y_min, x_max, y_max = tile.bounds_xy
    col = int(np.floor((float(x) - x_min) / (x_max - x_min) * float(width_px)))
    row = int(np.floor((y_max - float(y)) / (y_max - y_min) * float(height_px)))
    return row, col


def pixel_to_latlon(row: float, col: float, tile: LunarTile, width_px: int, height_px: int) -> tuple[float, float, float]:
    x, y = pixel_to_tile_xy(row, col, tile, width_px, height_px)
    lat, lon_180 = xy_to_latlon(x, y)
    return lat, lon_180, normalize_lon_360(lon_180)


def mask_centroid_latlon(mask_array: np.ndarray, tile: LunarTile) -> dict[str, float] | None:
    rows, cols = np.nonzero(mask_array)
    if rows.size == 0:
        return None
    row = float(rows.mean())
    col = float(cols.mean())
    lat, lon_180, lon_360 = pixel_to_latlon(row, col, tile, mask_array.shape[1], mask_array.shape[0])
    return {"centroid_row": row, "centroid_col": col, "centroid_lat": lat, "centroid_lon_180": lon_180, "centroid_lon_360": lon_360}


def summarize_mask_region(mask_array: np.ndarray, tile: LunarTile, datasets: dict[str, object] | None = None) -> dict[str, object]:
    rows, cols = np.nonzero(mask_array)
    centroid = mask_centroid_latlon(mask_array, tile)
    pixel_area = (tile.tile_size_m / mask_array.shape[1]) * (tile.tile_size_m / mask_array.shape[0])
    return {
        "tile_id": tile.tile_id,
        "area_pixels": int(rows.size),
        "area_m2_approx": float(rows.size * pixel_area),
        "bbox_pixels": [int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())] if rows.size else None,
        "centroid": centroid,
        "datasets": datasets or {},
    }
