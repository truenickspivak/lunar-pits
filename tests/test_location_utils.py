import unittest

from lunarpits.location import (
    collection_csv_url,
    location_output_dir_name,
    normalize_longitude_360,
    safe_location_label,
    select_diverse_nac_observations,
    volume_names,
)
from lunarpits.location.lroc_search import _query_footprint_geodataframe, score_nac_metadata_quality
from lunarpits.location.gather_location import find_existing_context_for_tile
from lunarpits.processing import normalize_product_id
from lunar_tile_pipeline.tiling import get_tile_for_latlon


class LocationUtilsTest(unittest.TestCase):
    def test_normalize_product_id_available_for_location_pipeline(self):
        self.assertEqual(normalize_product_id(" m107689129re.img "), "M107689129RE")

    def test_volume_name_generation_includes_suffixes(self):
        names = volume_names(max_volume=2)
        self.assertIn("LROLRC_0001", names)
        self.assertIn("LROLRC_0001C", names)
        self.assertIn("LROLRC_0002", names)

    def test_collection_csv_url_construction(self):
        self.assertEqual(
            collection_csv_url("LROLRC_0001"),
            "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-2-EDR-V1.0/LROLRC_0001/DATA/collection_lro-l-lroc-2-edr_lrolrc_0001_data.csv",
        )

    def test_output_folder_naming_from_lat_lon(self):
        self.assertEqual(location_output_dir_name(8.3355, 33.222), "8.335500_33.222000")

    def test_safe_location_label_for_batch_folder(self):
        self.assertEqual(safe_location_label("site 01/Tranquility"), "site_01_Tranquility")

    def test_existing_context_for_tile_detects_duplicate_folder(self):
        import json
        import tempfile
        from pathlib import Path

        tile = get_tile_for_latlon(14.2, 303.3, tile_size_km=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / "first_site"
            current = root / "second_site"
            existing.mkdir()
            current.mkdir()
            (existing / "tile_metadata.json").write_text(json.dumps({"tile_id": tile.tile_id}), encoding="utf-8")
            self.assertEqual(find_existing_context_for_tile(tile.tile_id, root, exclude=current), existing)

    def test_longitude_normalization(self):
        self.assertEqual(normalize_longitude_360(-56), 304)
        self.assertEqual(normalize_longitude_360(304), 304)

    def test_selection_prefers_contains_point_and_unique_products(self):
        try:
            import pandas as pd
        except ModuleNotFoundError:
            self.skipTest("pandas is not installed in this Python environment")
        df = pd.DataFrame(
            [
                {"product_id": "M000000003RE", "contains_point": None, "incidence_angle": None, "sun_azimuth": None},
                {"product_id": "M000000001RE", "contains_point": True, "incidence_angle": None, "sun_azimuth": None},
                {"product_id": "M000000001RE", "contains_point": True, "incidence_angle": None, "sun_azimuth": None},
                {"product_id": "M000000002RE", "contains_point": False, "incidence_angle": None, "sun_azimuth": None},
            ]
        )
        selected = select_diverse_nac_observations(df, max_products=2)
        self.assertEqual(selected["product_id"].tolist(), ["M000000001RE", "M000000003RE"])

    def test_point_inside_footprint_returns_product(self):
        gpd, Polygon = self._geo_deps()
        gdf = gpd.GeoDataFrame(
            [
                {
                    "PRODUCT_ID": "M000000001RE",
                    "NAC_FRM_ID": "RIGHT",
                    "URL": "https://example.invalid/M000000001RE.IMG",
                    "geometry": Polygon([(300, 8), (305, 8), (305, 12), (300, 12)]),
                }
            ],
            geometry="geometry",
        )
        df = _query_footprint_geodataframe(gdf, 10, -58, 0, "test")
        self.assertEqual(df["product_id"].tolist(), ["M000000001RE"])
        self.assertEqual(df["coverage_method"].tolist(), ["footprint_contains"])

    def test_point_outside_footprint_returns_no_product_even_with_center_metadata(self):
        gpd, Polygon = self._geo_deps()
        gdf = gpd.GeoDataFrame(
            [
                {
                    "PRODUCT_ID": "M000000001RE",
                    "CENTER_LAT": 10,
                    "CENTER_LON": 302,
                    "geometry": Polygon([(300, 8), (305, 8), (305, 12), (300, 12)]),
                }
            ],
            geometry="geometry",
        )
        df = _query_footprint_geodataframe(gdf, 40, -58, 0, "test")
        self.assertTrue(df.empty)

    def test_point_inside_footprint_far_from_center_still_returns_product(self):
        gpd, Polygon = self._geo_deps()
        gdf = gpd.GeoDataFrame(
            [
                {
                    "PRODUCT_ID": "M000000001LE",
                    "CENTER_LAT": 0,
                    "CENTER_LON": 0,
                    "geometry": Polygon([(300, 8), (305, 8), (305, 12), (300, 12)]),
                }
            ],
            geometry="geometry",
        )
        df = _query_footprint_geodataframe(gdf, 10, 302, 0, "test")
        self.assertEqual(df["product_id"].tolist(), ["M000000001LE"])

    def test_radius_intersection_includes_nearby_footprint(self):
        gpd, Polygon = self._geo_deps()
        gdf = gpd.GeoDataFrame(
            [
                {
                    "PRODUCT_ID": "M000000001RE",
                    "geometry": Polygon([(302.05, 8), (303, 8), (303, 9), (302.05, 9)]),
                }
            ],
            geometry="geometry",
        )
        df = _query_footprint_geodataframe(gdf, 8.5, 302, 5, "test")
        self.assertEqual(df["coverage_method"].tolist(), ["footprint_intersects_radius"])

    def test_edr_filter_converts_nac_c_footprints_and_excludes_wac(self):
        gpd, Polygon = self._geo_deps()
        geom = Polygon([(300, 8), (305, 8), (305, 12), (300, 12)])
        gdf = gpd.GeoDataFrame(
            [
                {"PRODUCT_ID": "M000000001RC", "geometry": geom},
                {"PRODUCT_ID": "W000000001CE", "geometry": geom},
                {"PRODUCT_ID": "M000000002RE", "geometry": geom},
            ],
            geometry="geometry",
        )
        df = _query_footprint_geodataframe(gdf, 10, 302, 0, "test")
        self.assertEqual(df["product_id"].tolist(), ["M000000001RE", "M000000002RE"])
        self.assertEqual(df["source_product_id"].tolist(), ["M000000001RC", "M000000002RE"])

    def test_selection_favors_diverse_angles_after_footprint_filter(self):
        try:
            import pandas as pd
        except ModuleNotFoundError:
            self.skipTest("pandas is not installed in this Python environment")
        df = pd.DataFrame(
            [
                {"product_id": "M000000001RE", "contains_point": True, "coverage_method": "footprint_contains", "incidence_angle": 10, "sun_azimuth": 10, "resolution": 1},
                {"product_id": "M000000002RE", "contains_point": True, "coverage_method": "footprint_contains", "incidence_angle": 11, "sun_azimuth": 11, "resolution": 1},
                {"product_id": "M000000003RE", "contains_point": True, "coverage_method": "footprint_contains", "incidence_angle": 80, "sun_azimuth": 200, "resolution": 1},
            ]
        )
        selected = select_diverse_nac_observations(df, max_products=2)
        self.assertEqual(selected["product_id"].tolist(), ["M000000001RE", "M000000003RE"])

    def test_metadata_quality_rejects_extreme_incidence(self):
        tier, score = score_nac_metadata_quality(
            coverage_method="footprint_contains",
            incidence_angle=145.0,
            emission_angle=10.0,
            phase_angle=150.0,
            resolution=0.5,
        )
        self.assertEqual(tier, "rejected_extreme_incidence")
        self.assertLess(score, 0)

    def test_selection_prefers_metadata_quality_before_extreme_diversity(self):
        try:
            import pandas as pd
        except ModuleNotFoundError:
            self.skipTest("pandas is not installed in this Python environment")
        df = pd.DataFrame(
            [
                {"product_id": "M000000001RE", "contains_point": True, "coverage_method": "footprint_contains", "quality_tier": "preferred", "quality_score": 120, "incidence_angle": 40, "sun_azimuth": 10, "resolution": 0.5},
                {"product_id": "M000000002RE", "contains_point": True, "coverage_method": "footprint_contains", "quality_tier": "preferred", "quality_score": 110, "incidence_angle": 45, "sun_azimuth": 20, "resolution": 0.5},
                {"product_id": "M000000003RE", "contains_point": True, "coverage_method": "footprint_contains", "quality_tier": "rejected_extreme_incidence", "quality_score": -20, "incidence_angle": 165, "sun_azimuth": 200, "resolution": 0.5},
            ]
        )
        selected = select_diverse_nac_observations(df, max_products=2)
        self.assertEqual(selected["product_id"].tolist(), ["M000000001RE", "M000000002RE"])

    def _geo_deps(self):
        try:
            import geopandas as gpd
            from shapely.geometry import Polygon
        except ModuleNotFoundError:
            self.skipTest("geopandas/shapely are not installed in this Python environment")
        return gpd, Polygon


if __name__ == "__main__":
    unittest.main()
