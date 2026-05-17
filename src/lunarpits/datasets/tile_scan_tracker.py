"""Sparse progress tracking for deterministic whole-Moon tile scanning."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from lunar_tile_pipeline.config import MOON_RADIUS_M
from lunar_tile_pipeline.projection import normalize_lon_180, normalize_lon_360
from lunarpits.tiling.ml_tiles import get_ml_tile_for_latlon, get_ml_tile_for_xy
from lunarpits.datasets.skylight_runner import PROJECT_ROOT, load_lroc_pit_catalog


DEFAULT_TRACKER_DB = PROJECT_ROOT / "data" / "moon_scan_progress.sqlite"
DEFAULT_PROGRESS_CSV = PROJECT_ROOT / "data" / "moon_scan_progress.csv"


@dataclass(frozen=True)
class GlobalGridSummary:
    tile_size_km: float
    tile_size_m: float
    min_tile_x: int
    max_tile_x: int
    min_tile_y: int
    max_tile_y: int
    total_rectangular_tiles: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "tile_size_km": self.tile_size_km,
            "tile_size_m": self.tile_size_m,
            "min_tile_x": self.min_tile_x,
            "max_tile_x": self.max_tile_x,
            "min_tile_y": self.min_tile_y,
            "max_tile_y": self.max_tile_y,
            "total_rectangular_tiles": self.total_rectangular_tiles,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def global_grid_summary(tile_size_km: float = 0.256) -> GlobalGridSummary:
    x_min = -math.pi * MOON_RADIUS_M
    x_max = math.pi * MOON_RADIUS_M
    y_min = -math.pi * MOON_RADIUS_M / 2.0
    y_max = math.pi * MOON_RADIUS_M / 2.0
    tile_size_m = float(tile_size_km) * 1000.0
    min_tile = get_ml_tile_for_xy(x_min, y_min, tile_size_m=tile_size_m)
    max_tile = get_ml_tile_for_xy(x_max, y_max, tile_size_m=tile_size_m)
    count_x = max_tile.tile_i - min_tile.tile_i + 1
    count_y = max_tile.tile_j - min_tile.tile_j + 1
    return GlobalGridSummary(
        tile_size_km=float(tile_size_km),
        tile_size_m=float(tile_size_km) * 1000.0,
        min_tile_x=min_tile.tile_i,
        max_tile_x=max_tile.tile_i,
        min_tile_y=min_tile.tile_j,
        max_tile_y=max_tile.tile_j,
        total_rectangular_tiles=count_x * count_y,
    )


def connect_tracker(db_path: str | Path = DEFAULT_TRACKER_DB) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tile_progress (
            tile_id TEXT PRIMARY KEY,
            tile_size_km REAL NOT NULL,
            tile_x INTEGER NOT NULL,
            tile_y INTEGER NOT NULL,
            center_lat REAL NOT NULL,
            center_lon_180 REAL NOT NULL,
            center_lon_360 REAL NOT NULL,
            status TEXT NOT NULL,
            label TEXT,
            label_source TEXT,
            location_id TEXT,
            source_name TEXT,
            source_lat REAL,
            source_lon_360 REAL,
            first_seen_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            notes TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tile_progress_status ON tile_progress(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tile_progress_label ON tile_progress(label)")
    conn.commit()


def initialize_tracker(
    db_path: str | Path = DEFAULT_TRACKER_DB,
    *,
    tile_size_km: float = 0.256,
    catalog_df: pd.DataFrame | None = None,
    refresh_catalog: bool = False,
) -> dict[str, int | float | str]:
    conn = connect_tracker(db_path)
    summary = global_grid_summary(tile_size_km)
    _set_metadata(conn, "grid_summary", json.dumps(summary.to_dict()))
    _set_metadata(conn, "projection", "spherical_equirectangular_moon_first_version")
    _set_metadata(conn, "moon_radius_m", str(MOON_RADIUS_M))
    _set_metadata(conn, "tracker_mode", "sparse_rows_only")
    if catalog_df is None:
        catalog_df = load_lroc_pit_catalog(refresh=refresh_catalog)
    seeded = seed_prechecked_lroc_pits(conn, catalog_df, tile_size_km=tile_size_km)
    conn.close()
    payload = summary.to_dict()
    payload["prechecked_positive_tiles"] = seeded
    payload["tracker_db"] = str(Path(db_path))
    return payload


def seed_prechecked_lroc_pits(conn: sqlite3.Connection, catalog_df: pd.DataFrame, *, tile_size_km: float = 0.256) -> int:
    count = 0
    for idx, row in catalog_df.iterrows():
        lat = _first_number(row, "Latitude", "Pit_Latitude")
        lon = _first_number(row, "Longitude_360", "Pit_Longitude", "Longitude")
        if lat is None or lon is None or not math.isfinite(lat) or not math.isfinite(lon):
            continue
        name = str(row.get("Name") or f"lroc_pit_{idx:04d}")
        location_id = f"positive_{idx:04d}_{_safe_name(name)}"
        tile = get_ml_tile_for_latlon(lat, lon, tile_size_m=float(tile_size_km) * 1000.0)
        upsert_tile_status(
            conn,
            tile.tile_i,
            tile.tile_j,
            tile_size_km=tile_size_km,
            status="prechecked_positive",
            label="positive_skylight_candidate",
            label_source="lroc_lunar_pit_locations_catalog",
            location_id=location_id,
            source_name=name,
            source_lat=float(lat),
            source_lon_360=normalize_lon_360(lon),
            notes="Seeded from official LROC Lunar Pit Locations catalog.",
        )
        count += 1
    conn.commit()
    return count


def upsert_tile_status(
    conn: sqlite3.Connection,
    tile_x: int,
    tile_y: int,
    *,
    tile_size_km: float = 0.256,
    status: str,
    label: str | None = None,
    label_source: str | None = None,
    location_id: str | None = None,
    source_name: str | None = None,
    source_lat: float | None = None,
    source_lon_360: float | None = None,
    notes: str | None = None,
) -> str:
    tile_size_m = float(tile_size_km) * 1000.0
    center_x = (int(tile_x) * tile_size_m) + tile_size_m / 2.0
    center_y = (int(tile_y) * tile_size_m) + tile_size_m / 2.0
    tile = get_ml_tile_for_xy(center_x, center_y, tile_size_m=tile_size_m)
    now = utc_now()
    conn.execute(
        """
        INSERT INTO tile_progress (
            tile_id, tile_size_km, tile_x, tile_y, center_lat, center_lon_180,
            center_lon_360, status, label, label_source, location_id, source_name,
            source_lat, source_lon_360, first_seen_at, updated_at, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(tile_id) DO UPDATE SET
            status=excluded.status,
            label=COALESCE(excluded.label, tile_progress.label),
            label_source=COALESCE(excluded.label_source, tile_progress.label_source),
            location_id=COALESCE(excluded.location_id, tile_progress.location_id),
            source_name=COALESCE(excluded.source_name, tile_progress.source_name),
            source_lat=COALESCE(excluded.source_lat, tile_progress.source_lat),
            source_lon_360=COALESCE(excluded.source_lon_360, tile_progress.source_lon_360),
            updated_at=excluded.updated_at,
            notes=COALESCE(excluded.notes, tile_progress.notes)
        """,
        (
            tile.tile_id,
            float(tile_size_km),
            tile.tile_i,
            tile.tile_j,
            tile.center_lat,
            tile.center_lon_180,
            tile.center_lon_360,
            status,
            label,
            label_source,
            location_id,
            source_name,
            source_lat,
            source_lon_360,
            now,
            now,
            notes,
        ),
    )
    conn.commit()
    return tile.tile_id


def export_progress_csv(db_path: str | Path = DEFAULT_TRACKER_DB, out_csv: str | Path = DEFAULT_PROGRESS_CSV) -> Path:
    conn = connect_tracker(db_path)
    rows = conn.execute("SELECT * FROM tile_progress ORDER BY tile_y, tile_x, tile_id").fetchall()
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = rows[0].keys()
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dict(row) for row in rows)
    else:
        out.write_text("tile_id,tile_size_km,tile_x,tile_y,status\n", encoding="utf-8")
    conn.close()
    return out


def progress_summary(db_path: str | Path = DEFAULT_TRACKER_DB) -> dict[str, int | dict[str, int] | str]:
    conn = connect_tracker(db_path)
    counts = {
        row["status"]: int(row["count"])
        for row in conn.execute("SELECT status, COUNT(*) AS count FROM tile_progress GROUP BY status ORDER BY status")
    }
    total_tracked = sum(counts.values())
    grid_summary = _get_metadata(conn, "grid_summary")
    conn.close()
    return {"tracked_tiles": total_tracked, "by_status": counts, "grid_summary": grid_summary or ""}


def _set_metadata(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def _get_metadata(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return str(row["value"]) if row else None


def _first_number(row: pd.Series, *names: str) -> float | None:
    for name in names:
        if name in row and not pd.isna(row[name]):
            try:
                return float(row[name])
            except (TypeError, ValueError):
                continue
    return None


def _safe_name(name: str) -> str:
    keep = [char if char.isalnum() else "_" for char in name.strip()]
    cleaned = "".join(keep).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "location"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize/export sparse deterministic Moon scan progress.")
    parser.add_argument("--db", default=str(DEFAULT_TRACKER_DB))
    parser.add_argument("--tile-size-km", type=float, default=0.256)
    parser.add_argument("--refresh-catalog", action="store_true")
    parser.add_argument("--export-csv", default=None, help="Export tracked rows to CSV after initialization.")
    parser.add_argument("--summary-only", action="store_true", help="Print existing tracker summary without seeding catalog rows.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.summary_only:
        payload = progress_summary(args.db)
    else:
        payload = initialize_tracker(args.db, tile_size_km=args.tile_size_km, refresh_catalog=args.refresh_catalog)
        payload["status_summary"] = progress_summary(args.db)
    if args.export_csv:
        payload["export_csv"] = str(export_progress_csv(args.db, args.export_csv))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
