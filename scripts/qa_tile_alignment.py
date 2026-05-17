"""Verify that processed NAC GeoTIFFs share the same tile coordinate frame."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import rasterio


MOON_RADIUS_M = 1737400.0


def _central_meridian(src: rasterio.DatasetReader) -> float:
    import re

    if src.crs is None:
        raise ValueError("Raster has no CRS")
    match = re.search(r'PARAMETER\["central_meridian",\s*([-+0-9.]+)\]', src.crs.to_wkt())
    if not match:
        raise ValueError("Could not find central_meridian in CRS WKT")
    return float(match.group(1))


def _latlon_to_raster_xy(src: rasterio.DatasetReader, lat: float, lon: float) -> tuple[float, float]:
    center_lon = _central_meridian(src)
    delta_lon = ((lon - center_lon + 180.0) % 360.0) - 180.0
    center_lat = _standard_parallel(src)
    x = MOON_RADIUS_M * math.radians(delta_lon)
    y = MOON_RADIUS_M * math.radians(lat - center_lat)
    return x, y


def _standard_parallel(src: rasterio.DatasetReader) -> float:
    import re

    if src.crs is None:
        raise ValueError("Raster has no CRS")
    match = re.search(r'PARAMETER\["standard_parallel_1",\s*([-+0-9.]+)\]', src.crs.to_wkt())
    return float(match.group(1)) if match else 0.0


def pixel_for_latlon(src: rasterio.DatasetReader, lat: float, lon: float) -> tuple[float, float]:
    x, y = _latlon_to_raster_xy(src, lat, lon)
    row, col = src.index(x, y)
    return float(row), float(col)


def qa_tile_alignment(tile_dir: str | Path, lat: float, lon: float) -> dict[str, object]:
    tile_path = Path(tile_dir)
    image_dir = tile_path / "nac" / "images"
    tif_paths = sorted(image_dir.glob("*.map.tif"))
    if not tif_paths:
        raise FileNotFoundError(f"No .map.tif files found in {image_dir}")

    rows = []
    reference = None
    for tif_path in tif_paths:
        with rasterio.open(tif_path) as src:
            row, col = pixel_for_latlon(src, lat, lon)
            info = {
                "path": str(tif_path),
                "width": src.width,
                "height": src.height,
                "bounds": [float(v) for v in src.bounds],
                "transform": [float(v) for v in src.transform[:6]],
                "target_row": row,
                "target_col": col,
            }
        if reference is None:
            reference = info
            info["matches_reference"] = True
        else:
            info["matches_reference"] = (
                info["width"] == reference["width"]
                and info["height"] == reference["height"]
                and info["bounds"] == reference["bounds"]
                and info["transform"] == reference["transform"]
                and info["target_row"] == reference["target_row"]
                and info["target_col"] == reference["target_col"]
            )
        rows.append(info)

    return {
        "tile_dir": str(tile_path),
        "target": {"lat": lat, "lon": lon},
        "all_match_reference": all(bool(row["matches_reference"]) for row in rows),
        "rasters": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that all tile GeoTIFFs map a coordinate to the same pixel.")
    parser.add_argument("tile_dir")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = qa_tile_alignment(args.tile_dir, args.lat, args.lon)
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["all_match_reference"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
