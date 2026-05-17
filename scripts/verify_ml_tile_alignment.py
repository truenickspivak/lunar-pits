"""Verify deterministic ML tile alignment and fixed scaling metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lunarpits.tiling.ml_tiles import pixel_in_tile  # noqa: E402


def load_metadata(paths: list[str]) -> list[dict[str, Any]]:
    payloads = []
    for item in paths:
        path = Path(item)
        if path.suffix.lower() == ".tif":
            path = path.with_suffix(".json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_metadata_file"] = str(path)
        payloads.append(payload)
    return payloads


def verify_alignment(lat: float, lon: float, metadata_paths: list[str]) -> dict[str, Any]:
    payloads = load_metadata(metadata_paths)
    if not payloads:
        raise ValueError("At least one metadata JSON or ML TIFF path is required.")

    expected = pixel_in_tile(
        lat,
        lon,
        tile_size_m=float(payloads[0]["tile_size_m"]),
        meters_per_pixel=float(payloads[0]["meters_per_pixel"]),
    )
    rows = []
    failures: list[str] = []
    baseline_keys = (
        "tile_i",
        "tile_j",
        "tile_size_m",
        "meters_per_pixel",
        "tile_size_px",
        "projection",
        "latitude_type",
        "longitude_direction",
        "longitude_domain",
        "center_latitude",
        "center_longitude",
        "normalization_policy",
        "normalization_constants",
        "preview_scaling_policy",
        "preview_scaling_constants",
    )
    baseline = {key: payloads[0].get(key) for key in baseline_keys}

    for payload in payloads:
        target = payload.get("target", {})
        row = {
            "source_nac_id": payload.get("source_nac_id"),
            "metadata_file": payload.get("_metadata_file"),
            "ml_tile_path": payload.get("ml_tile_path"),
            "ml_preview_path": payload.get("ml_preview_path"),
            "tile_i": payload.get("tile_i"),
            "tile_j": payload.get("tile_j"),
            "expected_tile_i": expected["tile_i"],
            "expected_tile_j": expected["tile_j"],
            "pixel_x": target.get("pixel_x"),
            "pixel_y": target.get("pixel_y"),
            "expected_pixel_x": expected["pixel_x"],
            "expected_pixel_y": expected["pixel_y"],
            "ul_x": payload.get("ul_x"),
            "ul_y": payload.get("ul_y"),
            "valid_pixel_fraction": payload.get("valid_pixel_fraction"),
            "normalization_policy": payload.get("normalization_policy"),
            "preview_scaling_constants": payload.get("preview_scaling_constants"),
            "ml_value_range": payload.get("ml_value_range"),
            "preview_png_value_range": payload.get("preview_png_value_range"),
        }
        rows.append(row)

        for key, value in baseline.items():
            if payload.get(key) != value:
                failures.append(f"{payload.get('source_nac_id')}: metadata mismatch for {key}: {payload.get(key)} != {value}")
        if payload.get("tile_i") != expected["tile_i"] or payload.get("tile_j") != expected["tile_j"]:
            failures.append(f"{payload.get('source_nac_id')}: tile index does not match deterministic coordinate mapping.")
        if abs(int(target.get("pixel_x", -999999)) - int(expected["pixel_x"])) > 1:
            failures.append(f"{payload.get('source_nac_id')}: pixel_x differs by more than one pixel.")
        if abs(int(target.get("pixel_y", -999999)) - int(expected["pixel_y"])) > 1:
            failures.append(f"{payload.get('source_nac_id')}: pixel_y differs by more than one pixel.")
        tile_size = float(payload.get("tile_size_m", 1.0))
        if abs(float(payload.get("ul_x", 0.0)) / tile_size - round(float(payload.get("ul_x", 0.0)) / tile_size)) > 1e-9:
            failures.append(f"{payload.get('source_nac_id')}: ul_x is not snapped to tile grid.")
        if abs(float(payload.get("ul_y", 0.0)) / tile_size - round(float(payload.get("ul_y", 0.0)) / tile_size)) > 1e-9:
            failures.append(f"{payload.get('source_nac_id')}: ul_y is not snapped to tile grid.")

    return {
        "input": {"lat": lat, "lon": lon},
        "expected_coordinate_mapping": expected,
        "results": rows,
        "passed": not failures,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify deterministic ML tile alignment sidecars.")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("metadata", nargs="+", help="Metadata JSON paths or ML TIFF paths")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = verify_alignment(args.lat, args.lon, args.metadata)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
