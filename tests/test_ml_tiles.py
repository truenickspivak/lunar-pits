import math
import unittest

import numpy as np

from lunarpits.tiling.ml_tiles import (
    MlScalingPolicy,
    apply_ml_normalization,
    get_ml_tile_for_latlon,
    make_ml_tile_id,
    pixel_in_tile,
    render_ml_preview,
    snap_bounds_to_tile_grid,
    validate_production_filename,
)


class MlTilesTest(unittest.TestCase):
    def test_same_coordinate_has_stable_tile_and_pixel(self):
        first = pixel_in_tile(8.3355, 33.222, tile_size_m=256.0, meters_per_pixel=1.0)
        second = pixel_in_tile(8.3355, 33.222, tile_size_m=256.0, meters_per_pixel=1.0)
        self.assertEqual(first["tile_i"], second["tile_i"])
        self.assertEqual(first["tile_j"], second["tile_j"])
        self.assertEqual(first["pixel_x"], second["pixel_x"])
        self.assertEqual(first["pixel_y"], second["pixel_y"])

    def test_grid_uses_floor_origin_bounds(self):
        tile = get_ml_tile_for_latlon(0.0, 0.0, tile_size_m=256.0, meters_per_pixel=1.0)
        self.assertEqual(tile.tile_i, 0)
        self.assertEqual(tile.tile_j, 0)
        self.assertEqual((tile.x_min_m, tile.y_min_m, tile.x_max_m, tile.y_max_m), (0.0, 0.0, 256.0, 256.0))
        self.assertEqual(tile.ul_x, 0.0)
        self.assertEqual(tile.ul_y, 256.0)

    def test_non_integer_tile_pixels_rejected(self):
        with self.assertRaises(ValueError):
            get_ml_tile_for_latlon(0.0, 0.0, tile_size_m=256.0, meters_per_pixel=1.3)

    def test_no_per_tile_normalization_in_preserve_float32(self):
        policy = MlScalingPolicy(normalization_policy="preserve_float32")
        low_range = np.array([[0.0, 1.0]], dtype=np.float32)
        high_range = np.array([[100.0, 101.0]], dtype=np.float32)
        np.testing.assert_array_equal(apply_ml_normalization(low_range, policy), low_range)
        np.testing.assert_array_equal(apply_ml_normalization(high_range, policy), high_range)

    def test_fixed_global_clip_maps_same_value_same_way(self):
        policy = MlScalingPolicy(normalization_policy="fixed_global_clip", ml_clip_min=0.0, ml_clip_max=100.0)
        a = apply_ml_normalization(np.array([[50.0, 90.0]], dtype=np.float32), policy)
        b = apply_ml_normalization(np.array([[-20.0, 50.0]], dtype=np.float32), policy)
        self.assertAlmostEqual(float(a[0, 0]), float(b[0, 1]))

    def test_preview_uses_fixed_constants(self):
        policy = MlScalingPolicy(normalization_policy="preserve_float32", preview_min=0.0, preview_max=100.0)
        dark_tile = render_ml_preview(np.array([[50.0]], dtype=np.float32), policy)
        bright_context_tile = render_ml_preview(np.array([[0.0, 50.0, 1000.0]], dtype=np.float32), policy)
        self.assertEqual(int(dark_tile[0, 0]), int(bright_context_tile[0, 1]))

    def test_forbidden_production_output_names(self):
        validate_production_filename("tile_x+000001_y+000002_M123456789LE_ml.tif")
        with self.assertRaises(ValueError):
            validate_production_filename("tile_x+000001_y+000002_M123456789LE_enhanced.png")

    def test_snap_bounds_to_grid(self):
        snapped = snap_bounds_to_tile_grid(1.0, 513.0, -1.0, 260.0, tile_size_m=256.0)
        self.assertEqual(snapped, (0.0, 768.0, -256.0, 512.0))
        for value in snapped:
            self.assertTrue(math.isclose(value / 256.0, round(value / 256.0)))

    def test_ml_tile_id_format(self):
        self.assertEqual(make_ml_tile_id(3935, 987), "tile_x+003935_y+000987")


if __name__ == "__main__":
    unittest.main()
