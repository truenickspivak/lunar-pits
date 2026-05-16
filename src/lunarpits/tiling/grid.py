"""Tile grid data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TileWindow:
    """A pixel window in row/column coordinates."""

    row_off: int
    col_off: int
    height: int
    width: int


def iter_tile_windows(
    raster_height: int,
    raster_width: int,
    *,
    tile_size: int,
    stride: int | None = None,
    drop_partial: bool = True,
) -> list[TileWindow]:
    """Build deterministic tile windows for a raster array.

    The function stays pure so it can be used after a GeoTIFF is opened by
    rasterio, PIL, or any other image reader.
    """
    if raster_height <= 0 or raster_width <= 0:
        raise ValueError("Raster height and width must be positive.")
    if tile_size <= 0:
        raise ValueError("Tile size must be positive.")

    stride = tile_size if stride is None else stride
    if stride <= 0:
        raise ValueError("Stride must be positive.")

    windows: list[TileWindow] = []
    for row in range(0, raster_height, stride):
        for col in range(0, raster_width, stride):
            height = min(tile_size, raster_height - row)
            width = min(tile_size, raster_width - col)
            if drop_partial and (height < tile_size or width < tile_size):
                continue
            windows.append(TileWindow(row, col, height, width))

    return windows
