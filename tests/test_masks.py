import numpy as np
import math

from lunar_tile_pipeline.masks import mask_centroid_latlon, pixel_to_latlon, pixel_to_tile_xy
from lunar_tile_pipeline.tiling import get_tile_for_latlon


def test_center_pixel_converts_to_tile_center_latlon():
    tile = get_tile_for_latlon(0, 0, tile_size_km=10)
    lat, lon_180, lon_360 = pixel_to_latlon(511.5, 511.5, tile, 1024, 1024)
    assert math.isclose(lat, tile.center_lat)
    assert math.isclose(lon_180, tile.center_lon_180)
    assert math.isclose(lon_360, tile.center_lon_360)


def test_corner_pixels_convert_near_tile_bounds():
    tile = get_tile_for_latlon(0, 0, tile_size_km=10)
    x, y = pixel_to_tile_xy(0, 0, tile, 1024, 1024)
    assert abs(x - tile.bounds_xy[0]) <= 10
    assert abs(y - tile.bounds_xy[3]) <= 10


def test_mask_centroid_latlon():
    tile = get_tile_for_latlon(0, 0, tile_size_km=10)
    mask = np.zeros((1024, 1024), dtype=bool)
    mask[510:514, 510:514] = True
    centroid = mask_centroid_latlon(mask, tile)
    assert centroid is not None
    assert abs(centroid["centroid_lat"]) <= 0.01
    assert abs(centroid["centroid_lon_180"]) <= 0.01
