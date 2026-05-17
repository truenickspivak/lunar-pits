import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from lunarpits.datasets.skylight_runner import (
    DATASET_QUEUE_COLUMNS,
    DatasetRunConfig,
    _catalog_product_ids,
    audit_positive_coordinates,
    build_dataset_queue,
    load_lroc_pit_catalog,
    run_dataset,
    validate_location_completion,
)


class SkylightDatasetRunnerTest(unittest.TestCase):
    def test_lroc_pit_catalog_loader_handles_cp1252(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pits.csv"
            path.write_text("Name,Latitude,Longitude_360,Description\nMare Pit,1,2,depth \u2013 known\n", encoding="cp1252")
            df = load_lroc_pit_catalog(path)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.attrs["encoding"], "cp1252")

    def test_queue_is_ordered_positive_negative_undetermined(self):
        catalog = pd.DataFrame(
            [
                {"Name": "Pit A", "Latitude": 1.0, "Longitude_360": 2.0, "Terrain": "Mare"},
                {"Name": "Pit B", "Latitude": -1.0, "Longitude_360": 200.0, "Terrain": "Highland"},
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            config = DatasetRunConfig(
                queue_csv=Path(tmp) / "queue.csv",
                catalog_cache=Path(tmp) / "pits.csv",
                negative_multiplier=2,
                undetermined_multiplier=1,
            )
            with patch("lunarpits.datasets.skylight_runner.load_lroc_pit_catalog", return_value=catalog):
                queue = build_dataset_queue(config)
        self.assertEqual(queue["split_group"].tolist(), ["positive", "positive", "negative", "negative", "negative", "negative", "undetermined", "undetermined"])
        self.assertEqual(queue["status"].unique().tolist(), ["pending"])
        self.assertEqual(list(queue.columns), DATASET_QUEUE_COLUMNS)

    def test_positive_queue_preserves_catalog_image_hints(self):
        catalog = pd.DataFrame(
            [
                {
                    "Name": "Marius Hills Pit",
                    "Latitude": 14.0,
                    "Longitude_360": 303.0,
                    "Image 1": "M155607349R",
                    "Image 2": "M122584310L",
                    "Image 3": "M1256963084R",
                    "Stereo IDs": "M1274616922L,M1274609891L",
                }
            ]
        )
        self.assertEqual(
            _catalog_product_ids(catalog.iloc[0]),
            ["M155607349RE", "M122584310LE", "M1256963084RE", "M1274616922LE", "M1274609891LE"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            config = DatasetRunConfig(
                queue_csv=Path(tmp) / "queue.csv",
                catalog_cache=Path(tmp) / "pits.csv",
                negative_multiplier=0,
                undetermined_multiplier=0,
            )
            with patch("lunarpits.datasets.skylight_runner.load_lroc_pit_catalog", return_value=catalog):
                queue = build_dataset_queue(config)
        self.assertIn("M155607349RE", queue.loc[0, "preferred_product_ids"])

    def test_dry_run_creates_queue_but_leaves_pending(self):
        catalog = pd.DataFrame([{"Name": "Pit A", "Latitude": 1.0, "Longitude_360": 2.0}])
        with tempfile.TemporaryDirectory() as tmp:
            config = DatasetRunConfig(
                queue_csv=Path(tmp) / "dataset_queue.csv",
                positive_audit_csv=Path(tmp) / "audit.csv",
                run_summary_json=Path(tmp) / "summary.json",
                catalog_cache=Path(tmp) / "pits.csv",
                negative_multiplier=1,
                undetermined_multiplier=1,
                dry_run=True,
            )
            with patch("lunarpits.datasets.skylight_runner.load_lroc_pit_catalog", return_value=catalog):
                queue = run_dataset(config, gatherer=self._failing_gatherer)
            self.assertTrue(config.queue_csv.exists())
            self.assertTrue(config.positive_audit_csv.exists())
            self.assertTrue(config.run_summary_json.exists())
            self.assertEqual(set(queue["status"]), {"pending"})

    def test_preflight_fails_shifted_positive_coordinate(self):
        catalog = pd.DataFrame([{"Name": "loc 0", "Latitude": 1.0, "Longitude_360": 2.0}])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = DatasetRunConfig(queue_csv=root / "queue.csv", positive_audit_csv=root / "audit.csv")
            queue = pd.DataFrame([self._queue_row(0, "pending", root)], columns=DATASET_QUEUE_COLUMNS)
            queue.loc[0, "lat"] = 1.05
            with patch("lunarpits.datasets.skylight_runner.load_lroc_pit_catalog", return_value=catalog):
                with self.assertRaises(RuntimeError):
                    audit_positive_coordinates(queue, config)

    def test_preflight_passes_current_positive_coordinate(self):
        catalog = pd.DataFrame([{"Name": "loc 0", "Latitude": 1.0, "Longitude_360": 2.0}])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = DatasetRunConfig(queue_csv=root / "queue.csv", positive_audit_csv=root / "audit.csv")
            queue = pd.DataFrame([self._queue_row(0, "pending", root)], columns=DATASET_QUEUE_COLUMNS)
            with patch("lunarpits.datasets.skylight_runner.load_lroc_pit_catalog", return_value=catalog):
                audit = audit_positive_coordinates(queue, config)
            self.assertEqual(audit.loc[0, "audit_status"], "passed")

    def test_resume_skips_complete_and_updates_failed_after_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = pd.DataFrame(
                [
                    self._queue_row(0, "complete", root),
                    self._queue_row(1, "pending", root),
                ],
                columns=DATASET_QUEUE_COLUMNS,
            )
            queue_csv = root / "dataset_queue.csv"
            queue.to_csv(queue_csv, index=False)
            config = DatasetRunConfig(queue_csv=queue_csv, positive_audit_csv=root / "audit.csv", run_summary_json=root / "summary.json", max_rows=1)
            catalog = pd.DataFrame(
                [
                    {"Name": "loc 0", "Latitude": 1.0, "Longitude_360": 2.0},
                    {"Name": "loc 1", "Latitude": 1.0, "Longitude_360": 2.0},
                ]
            )
            with patch("lunarpits.datasets.skylight_runner.load_lroc_pit_catalog", return_value=catalog):
                result = run_dataset(config, gatherer=self._failing_gatherer)
            self.assertEqual(result.loc[0, "status"], "complete")
            self.assertEqual(result.loc[1, "status"], "failed")
            self.assertIn("boom", result.loc[1, "error"])

    def test_resume_marks_complete_and_writes_manifest_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = pd.DataFrame([self._queue_row(0, "pending", root)], columns=DATASET_QUEUE_COLUMNS)
            queue_csv = root / "dataset_queue.csv"
            queue.to_csv(queue_csv, index=False)
            config = DatasetRunConfig(queue_csv=queue_csv, positive_audit_csv=root / "audit.csv", run_summary_json=root / "summary.json", max_rows=1, dry_run=False)
            catalog = pd.DataFrame([{"Name": "loc 0", "Latitude": 1.0, "Longitude_360": 2.0}])
            with patch("lunarpits.datasets.skylight_runner.load_lroc_pit_catalog", return_value=catalog):
                result = run_dataset(config, gatherer=lambda **kwargs: self._fake_context(root))
            self.assertEqual(result.loc[0, "status"], "complete")
            manifest = root / "loc_0" / "manifest.csv"
            annotations = root / "loc_0" / "annotations.csv"
            self.assertTrue(manifest.exists())
            self.assertTrue(annotations.exists())
            manifest_df = pd.read_csv(manifest)
            self.assertIn("label", manifest_df.columns)
            self.assertIn("gravity_available", manifest_df.columns)

    def test_completion_validation_fails_missing_required_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = DatasetRunConfig(queue_csv=root / "queue.csv")
            with self.assertRaises(RuntimeError):
                validate_location_completion(root, {}, config)

    def test_completion_validation_rejects_black_tiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nac" / "masks").mkdir(parents=True)
            (root / "nac" / "images").mkdir(parents=True)
            (root / "tile.json").write_text("{}", encoding="utf-8")
            (root / "annotations.csv").write_text("label\npositive\n", encoding="utf-8")
            (root / "nac" / "masks" / "README.md").write_text("masks\n", encoding="utf-8")
            image_path = root / "nac" / "images" / "M000000001RE.tif"
            image_path.write_text("fake", encoding="utf-8")
            sidecar_path = image_path.with_suffix(".json")
            sidecar_path.write_text('{"valid_pixel_fraction": 0.0}', encoding="utf-8")
            pd.DataFrame([{"image_path": str(image_path), "metadata_path": str(sidecar_path)}]).to_csv(root / "manifest.csv", index=False)
            config = DatasetRunConfig(queue_csv=root / "queue.csv")
            with self.assertRaises(RuntimeError):
                validate_location_completion(root, {}, config)

    def _queue_row(self, index, status, root):
        values = {column: "" for column in DATASET_QUEUE_COLUMNS}
        values.update(
            {
                "queue_index": index,
                "split_group": "positive",
                "label": "positive_skylight_candidate",
                "label_source": "test",
                "name": f"loc {index}",
                "lat": 1.0,
                "lon_180": 2.0,
                "lon_360": 2.0,
                "location_id": f"loc_{index}",
                "status": status,
                "num_nac_requested": 5,
                "num_nac_completed": 0,
                "location_path": str(root / f"loc_{index}"),
                "tile_json": str(root / f"loc_{index}" / "tile.json"),
                "source_row": index,
            }
        )
        return values

    def _fake_context(self, root):
        location = root / "loc_0"
        location.mkdir(parents=True, exist_ok=True)
        images = location / "nac" / "images"
        images.mkdir(parents=True, exist_ok=True)
        (images / "M000000001RE.tif").write_text("fake", encoding="utf-8")
        (images / "M000000001RE.png").write_text("fake", encoding="utf-8")
        (images / "M000000001RE.json").write_text('{"valid_pixel_fraction": 1.0}', encoding="utf-8")
        return {
            "output_dir": str(location),
            "tile": {"tile_id": "moon_256m_x+000000_y+000000"},
            "gravity": {"available": True},
            "topology": {"available": True},
            "ir": {"available": False},
            "lroc_nac": {
                "processed": [
                    {
                        "product_id": "M000000001RE",
                        "context_tif": str(location / "nac" / "images" / "M000000001RE.tif"),
                        "quicklook": str(location / "nac" / "images" / "M000000001RE.png"),
                    }
                ]
            },
        }

    def _failing_gatherer(self, **kwargs):
        raise RuntimeError("boom")


if __name__ == "__main__":
    unittest.main()
