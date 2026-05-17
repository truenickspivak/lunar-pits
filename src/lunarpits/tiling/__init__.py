"""Raster tiling primitives for training and inference."""

from lunarpits.tiling.grid import TileWindow, iter_tile_windows
from lunarpits.tiling.ml_tiles import (
    MlScalingPolicy,
    MlTileSpec,
    apply_ml_normalization,
    get_ml_tile_for_latlon,
    latlon_to_global_xy,
    pixel_in_tile,
    render_ml_preview,
)

__all__ = [
    "TileWindow",
    "iter_tile_windows",
    "MlScalingPolicy",
    "MlTileSpec",
    "apply_ml_normalization",
    "get_ml_tile_for_latlon",
    "latlon_to_global_xy",
    "pixel_in_tile",
    "render_ml_preview",
]
