"""Input/output helpers for tile products."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from lunar_tile_pipeline.projection import normalize_lon_180, normalize_lon_360
from lunar_tile_pipeline.tiling import LunarTile


def tile_output_dir(tile: LunarTile, out_dir: str | Path) -> Path:
    return Path(out_dir) / tile.tile_id


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_tile_metadata(
    tile: LunarTile,
    out_dir: str | Path,
    *,
    input_query: dict[str, Any] | None = None,
    data_sources: dict[str, Any] | None = None,
) -> Path:
    folder = tile_output_dir(tile, out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    metadata = tile.to_metadata()
    if input_query:
        metadata["input_query"] = {
            **input_query,
            "lon_180": normalize_lon_180(input_query["lon_original"]),
            "lon_360": normalize_lon_360(input_query["lon_original"]),
        }
    metadata["data_sources"] = data_sources or {}
    path = folder / "tile_metadata.json"
    write_json(path, metadata)
    return path


def write_products(tile: LunarTile, out_dir: str | Path, products: pd.DataFrame) -> tuple[Path, Path]:
    folder = tile_output_dir(tile, out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "lroc_nac_edr_products.csv"
    json_path = folder / "lroc_nac_edr_products.json"
    products.to_csv(csv_path, index=False)
    json_path.write_text(products.to_json(orient="records", indent=2), encoding="utf-8")
    return csv_path, json_path
