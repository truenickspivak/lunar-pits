from itertools import islice

import math

from lunar_tile_pipeline.projection import latlon_to_xy
from lunar_tile_pipeline.tiling import get_tile_by_indices, get_tile_for_latlon, get_tile_for_xy, iter_global_tiles


def test_tile_center_at_origin():
    tile = get_tile_for_latlon(0, 0, tile_size_km=10)
    assert tile.tile_x == 0
    assert tile.tile_y == 0
    assert tile.center_x == 0
    assert tile.center_y == 0
    assert all(math.isclose(a, b) for a, b in zip(tile.bounds_xy, (-5000, -5000, 5000, 5000)))


def test_stable_tile_assignment():
    a = get_tile_for_latlon(14.123, 303.456, tile_size_km=10)
    b = get_tile_for_latlon(14.123, -56.544, tile_size_km=10)
    assert a.tile_id == b.tile_id


def test_boundary_behavior():
    assert get_tile_for_xy(4999.9, 0, tile_size_km=10).tile_x == 0
    assert get_tile_for_xy(5000.1, 0, tile_size_km=10).tile_x == 1
    assert get_tile_for_xy(-5000.1, 0, tile_size_km=10).tile_x == -1


def test_iter_global_tiles_nonzero_and_valid_centers():
    tiles = list(islice(iter_global_tiles(tile_size_km=1000), 20))
    assert tiles
    for tile in tiles:
        assert -90 <= tile.center_lat <= 90
        assert -180 <= tile.center_lon_180 < 180


def test_get_tile_by_indices_round_trip_center():
    tile = get_tile_by_indices(123, -45, tile_size_km=10)
    x, y = latlon_to_xy(tile.center_lat, tile.center_lon_180)
    assert math.isclose(x, tile.center_x)
    assert math.isclose(y, tile.center_y)
