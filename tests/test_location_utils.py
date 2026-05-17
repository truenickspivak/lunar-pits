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
from lunarpits.location.gather_location import (
    _prioritize_preferred_products,
    _remove_empty_duplicate_shell,
    _validate_crop_coordinate_lock,
    _write_or_validate_coordinate_lock,
    _remove_legacy_context_dirs,
    _tile_training_payload,
    find_existing_context_for_tile,
)
from lunarpits.location.gather_location import _write_training_manifests, production_tile_for_latlon
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

    def test_production_location_tile_matches_ml_crop_grid(self):
        tile = production_tile_for_latlon(-10.248433, 15.702750, tile_size_km=0.256, meters_per_pixel=0.5)
        self.assertEqual(tile.tile_id, "tile_x+001859_y-001214")
        self.assertEqual(tile.tile_x, 1859)
        self.assertEqual(tile.tile_y, -1214)
        self.assertEqual(tile.to_metadata()["ml_tile"]["tile_id"], tile.tile_id)

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

    def test_tile_training_payload_embeds_context_sources(self):
        payload = _tile_training_payload(
            {
                "target": {"lat": 1, "lon": 2},
                "tile": {"tile_id": "moon_0p256km_x+000000_y+000000"},
                "lroc_nac": {"num_selected": 5, "tile_method": "post-map-crop"},
                "topology": {"available": True, "dataset": "LOLA"},
                "gravity": {"available": True, "dataset": "GRAIL"},
                "ir": {"available": False, "dataset": "Diviner"},
                "warnings": ["test"],
            }
        )
        self.assertEqual(payload["tile"]["tile_id"], "moon_0p256km_x+000000_y+000000")
        self.assertEqual(payload["context"]["topology"]["dataset"], "LOLA")
        self.assertEqual(payload["selection"]["tile_method"], "post-map-crop")

    def test_remove_legacy_context_dirs(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("topology", "gravity", "ir"):
                folder = root / name
                folder.mkdir()
                (folder / "context.json").write_text("{}", encoding="utf-8")
            _remove_legacy_context_dirs(root)
            self.assertFalse((root / "topology").exists())
            self.assertFalse((root / "gravity").exists())
            self.assertFalse((root / "ir").exists())

    def test_remove_empty_duplicate_shell_only_removes_empty_folders(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty_location" / "nac" / "images"
            empty.mkdir(parents=True)
            _remove_empty_duplicate_shell(root / "empty_location")
            self.assertFalse((root / "empty_location").exists())

            nonempty = root / "nonempty_location"
            nonempty.mkdir()
            (nonempty / "tile.json").write_text("{}", encoding="utf-8")
            _remove_empty_duplicate_shell(nonempty)
            self.assertTrue(nonempty.exists())

    def test_coordinate_lock_refuses_reused_folder_for_shifted_coordinate(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tile = production_tile_for_latlon(8.3355, 33.222, tile_size_km=0.256, meters_per_pixel=1.0)
            _write_or_validate_coordinate_lock(root, lat=8.3355, lon=33.222, tile=tile, force=False)
            shifted = production_tile_for_latlon(8.34, 33.222, tile_size_km=0.256, meters_per_pixel=1.0)
            with self.assertRaises(ValueError):
                _write_or_validate_coordinate_lock(root, lat=8.34, lon=33.222, tile=shifted, force=False)

    def test_crop_coordinate_lock_rejects_wrong_sidecar_target(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "M000000001RE.json"
            expected_tile = production_tile_for_latlon(8.3355, 33.222, tile_size_km=0.256, meters_per_pixel=1.0)
            sidecar.write_text(
                json.dumps(
                    {
                        "tile_id": expected_tile.tile_id,
                        "tile_i": expected_tile.tile_x,
                        "tile_j": expected_tile.tile_y,
                        "target": {"lat": 8.0, "lon_original": 33.222, "pixel_x": 0, "pixel_y": 0},
                    }
                ),
                encoding="utf-8",
            )
            result = _validate_crop_coordinate_lock(sidecar, lat=8.3355, lon=33.222, tile_size_m=256.0, meters_per_pixel=1.0)
            self.assertFalse(result["passed"])

    def test_write_training_manifests_for_ad_hoc_location(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = {
                "target": {"lat": -10.248433, "lon": 15.70275},
                "tile": {"tile_id": "tile_x+001859_y-001214"},
                "lroc_nac": {
                    "processed": [
                        {
                            "product_id": "M000000001RE",
                            "context_tif": str(root / "nac" / "images" / "M000000001RE.tif"),
                            "quicklook": str(root / "nac" / "images" / "M000000001RE.png"),
                        }
                    ]
                },
                "gravity": {"available": True},
                "topology": {"available": True},
                "ir": {"available": False},
            }
            _write_training_manifests(root, context)
            self.assertTrue((root / "manifest.csv").exists())
            self.assertTrue((root / "annotations.csv").exists())
            self.assertTrue((root / "nac" / "masks" / "README.md").exists())

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

    def test_preferred_products_are_boosted_after_footprint_search(self):
        try:
            import pandas as pd
        except ModuleNotFoundError:
            self.skipTest("pandas is not installed in this Python environment")
        df = pd.DataFrame(
            [
                {"product_id": "M000000001RE", "rank": 1},
                {"product_id": "M000000002RE", "rank": 2},
                {"product_id": "M000000003RE", "rank": 3},
            ]
        )
        selected = _prioritize_preferred_products(df, ["M000000003R"], max_products=3)
        self.assertEqual(selected["product_id"].tolist(), ["M000000003RE", "M000000001RE", "M000000002RE"])

    def _geo_deps(self):
        try:
            import geopandas as gpd
            from shapely.geometry import Polygon
        except ModuleNotFoundError:
            self.skipTest("geopandas/shapely are not installed in this Python environment")
        return gpd, Polygon


if __name__ == "__main__":
    unittest.main()
