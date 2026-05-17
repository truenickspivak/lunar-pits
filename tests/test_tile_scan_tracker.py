import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from lunarpits.datasets.tile_scan_tracker import (
    export_progress_csv,
    global_grid_summary,
    initialize_tracker,
    progress_summary,
    seed_prechecked_lroc_pits,
    upsert_tile_status,
)


class TileScanTrackerTest(unittest.TestCase):
    def test_global_grid_summary_is_large_but_not_enumerated(self):
        summary = global_grid_summary(tile_size_km=0.256)
        self.assertGreater(summary.total_rectangular_tiles, 100_000_000)
        self.assertLess(summary.min_tile_x, summary.max_tile_x)
        self.assertLess(summary.min_tile_y, summary.max_tile_y)

    def test_seed_prechecked_lroc_pits_tracks_deterministic_tile_once(self):
        catalog = pd.DataFrame(
            [
                {"Name": "Pit A", "Latitude": 8.3355, "Longitude_360": 33.222},
                {"Name": "Pit A duplicate same tile", "Latitude": 8.3355001, "Longitude_360": 33.2220001},
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            conn = sqlite3.connect(Path(tmp) / "progress.sqlite")
            conn.row_factory = sqlite3.Row
            from lunarpits.datasets.tile_scan_tracker import ensure_schema

            ensure_schema(conn)
            seeded = seed_prechecked_lroc_pits(conn, catalog, tile_size_km=0.256)
            rows = conn.execute("SELECT * FROM tile_progress").fetchall()
            self.assertEqual(seeded, 2)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "prechecked_positive")
            conn.close()

    def test_initialize_and_export_progress_csv(self):
        catalog = pd.DataFrame([{"Name": "Pit A", "Latitude": 8.3355, "Longitude_360": 33.222}])
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "progress.sqlite"
            out_csv = Path(tmp) / "progress.csv"
            payload = initialize_tracker(db, tile_size_km=0.256, catalog_df=catalog)
            self.assertEqual(payload["prechecked_positive_tiles"], 1)
            summary = progress_summary(db)
            self.assertEqual(summary["by_status"]["prechecked_positive"], 1)
            exported = export_progress_csv(db, out_csv)
            self.assertTrue(exported.exists())
            self.assertIn("prechecked_positive", exported.read_text(encoding="utf-8"))

    def test_upsert_tile_status_marks_completed_tile(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "progress.sqlite"
            from lunarpits.datasets.tile_scan_tracker import connect_tracker

            conn = connect_tracker(db)
            tile_id = upsert_tile_status(conn, 1, 2, tile_size_km=0.256, status="complete_no_detection", label="negative_auto")
            self.assertIn("tile_x+000001_y+000002", tile_id)
            row = conn.execute("SELECT status, label FROM tile_progress WHERE tile_id=?", (tile_id,)).fetchone()
            self.assertEqual(row["status"], "complete_no_detection")
            self.assertEqual(row["label"], "negative_auto")
            conn.close()


if __name__ == "__main__":
    unittest.main()
