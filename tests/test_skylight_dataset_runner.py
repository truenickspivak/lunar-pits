import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from lunarpits.datasets.skylight_runner import (
    DATASET_QUEUE_COLUMNS,
    DatasetRunConfig,
    build_dataset_queue,
    load_lroc_pit_catalog,
    run_dataset,
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

    def test_dry_run_creates_queue_but_leaves_pending(self):
        catalog = pd.DataFrame([{"Name": "Pit A", "Latitude": 1.0, "Longitude_360": 2.0}])
        with tempfile.TemporaryDirectory() as tmp:
            config = DatasetRunConfig(
                queue_csv=Path(tmp) / "dataset_queue.csv",
                catalog_cache=Path(tmp) / "pits.csv",
                negative_multiplier=1,
                undetermined_multiplier=1,
                dry_run=True,
            )
            with patch("lunarpits.datasets.skylight_runner.load_lroc_pit_catalog", return_value=catalog):
                queue = run_dataset(config, gatherer=self._failing_gatherer)
            self.assertTrue(config.queue_csv.exists())
            self.assertEqual(set(queue["status"]), {"pending"})

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
            config = DatasetRunConfig(queue_csv=queue_csv, max_rows=1)
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
            config = DatasetRunConfig(queue_csv=queue_csv, max_rows=1, dry_run=False)
            result = run_dataset(config, gatherer=lambda **kwargs: self._fake_context(root))
            self.assertEqual(result.loc[0, "status"], "complete")
            manifest = root / "loc_0" / "manifest.csv"
            annotations = root / "loc_0" / "annotations.csv"
            self.assertTrue(manifest.exists())
            self.assertTrue(annotations.exists())
            manifest_df = pd.read_csv(manifest)
            self.assertIn("label", manifest_df.columns)
            self.assertIn("gravity_available", manifest_df.columns)

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
            }
        )
        return values

    def _fake_context(self, root):
        location = root / "loc_0"
        location.mkdir(parents=True, exist_ok=True)
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
                        "context_tif": str(location / "nac" / "images" / "tile_M000000001RE_ml.tif"),
                        "quicklook": str(location / "nac" / "images" / "tile_M000000001RE_ml_preview.png"),
                    }
                ]
            },
        }

    def _failing_gatherer(self, **kwargs):
        raise RuntimeError("boom")


if __name__ == "__main__":
    unittest.main()
