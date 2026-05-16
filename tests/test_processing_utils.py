from pathlib import PureWindowsPath
import unittest

from lunarpits.processing import normalize_product_id, windows_path_to_wsl
from lunarpits.processing.lroc_product import build_processing_plan, parse_caminfo_center, tile_map_bounds_from_latlon


class ProcessingUtilsTest(unittest.TestCase):
    def test_normalize_product_id_strips_case_and_known_suffixes(self):
        self.assertEqual(normalize_product_id(" m114328462re.img "), "M114328462RE")
        self.assertEqual(normalize_product_id("M114328462RE.XML"), "M114328462RE")

    def test_normalize_product_id_leaves_core_id_unchanged(self):
        self.assertEqual(normalize_product_id("M114328462RE"), "M114328462RE")

    def test_windows_path_to_wsl_converts_absolute_path(self):
        path = PureWindowsPath(r"C:\Users\nicks\Desktop\lunar-pits\data\out.tif")
        self.assertEqual(
            windows_path_to_wsl(path),
            "/mnt/c/Users/nicks/Desktop/lunar-pits/data/out.tif",
        )

    def test_windows_path_to_wsl_requires_drive(self):
        with self.assertRaises(ValueError):
            windows_path_to_wsl(r"\Users\nicks\data.tif")

    def test_parse_caminfo_center_reads_center_fields(self):
        caminfo = """
        Object = Geometry
          CenterLatitude  = 8.3355
          CenterLongitude = 33.222
        End_Object
        """
        self.assertEqual(parse_caminfo_center(caminfo), (8.3355, 33.222))

    def test_tile_map_bounds_from_latlon_is_deterministic(self):
        first = tile_map_bounds_from_latlon(14.2, 303.3, 1)
        second = tile_map_bounds_from_latlon(14.2, 303.3, 1)
        self.assertEqual(first, second)
        self.assertEqual(first.tile_id, "moon_1km_x-001719_y+000431")
        self.assertAlmostEqual(first.min_lat, 14.19698001633587)
        self.assertAlmostEqual(first.max_lon, 303.32750253641535)

    def test_tile_processing_plan_uses_separate_output_folder(self):
        tile_map = tile_map_bounds_from_latlon(14.2, 303.3, 1)
        plan = build_processing_plan(
            "M170932152LE",
            center_lat=None,
            center_lon=None,
            pixel_resolution=0.5,
            tile_map=tile_map,
        )
        self.assertIn(r"data\processed\lroc_tile_tif\moon_1km_x-001719_y+000431", str(plan.final_tif_win))
        self.assertEqual(plan.center_lat, tile_map.center_lat)

    def test_tile_processing_plan_accepts_output_dir_override(self):
        tile_map = tile_map_bounds_from_latlon(14.2, 303.3, 1)
        plan = build_processing_plan(
            "M170932152LE",
            center_lat=None,
            center_lon=None,
            pixel_resolution=0.5,
            tile_map=tile_map,
            output_dir=PureWindowsPath(r"C:\tmp\location\crops"),
        )
        self.assertEqual(str(plan.final_tif_win), r"C:\tmp\location\crops\M170932152LE.map.tif")


if __name__ == "__main__":
    unittest.main()
