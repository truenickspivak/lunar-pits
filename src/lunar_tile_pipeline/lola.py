"""LOLA topography sampling placeholder."""

from __future__ import annotations

from pathlib import Path

from lunar_tile_pipeline.tiling import LunarTile


def sample_dataset_for_tile(tile: LunarTile, dataset_path: str | Path | None, output_dir: str | Path) -> dict[str, object]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if dataset_path is None:
        return {"available": False, "dataset": "LOLA", "tile_id": tile.tile_id, "reason": "No LOLA raster configured yet"}
    return {"available": False, "dataset": "LOLA", "tile_id": tile.tile_id, "source_file": str(dataset_path), "todo": "Implement raster sampling/resampling for this tile."}
