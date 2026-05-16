import math

from lunar_tile_pipeline.projection import latlon_to_xy, xy_to_latlon


def test_projection_round_trip():
    lat, lon = 14.123, 303.456
    x, y = latlon_to_xy(lat, lon)
    out_lat, out_lon = xy_to_latlon(x, y)
    assert math.isclose(out_lat, lat)
    assert math.isclose(out_lon, -56.544)
