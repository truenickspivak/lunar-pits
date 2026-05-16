from lunar_tile_pipeline.projection import normalize_lon_180, normalize_lon_360


def test_normalize_lon_180():
    assert normalize_lon_180(303) == -57
    assert normalize_lon_180(360) == 0
    assert normalize_lon_180(-180) == -180


def test_normalize_lon_360():
    assert normalize_lon_360(-57) == 303
    assert normalize_lon_360(360) == 0
    assert normalize_lon_360(303) == 303
