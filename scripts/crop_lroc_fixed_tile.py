"""Create a production ML tile from an already map-projected LROC TIFF.

This is the efficient path: run ISIS cam2map once for the full NAC
observation, then reproject/crop the deterministic global tile afterward.
The saved PNG is an ML preview, not a browse image: it uses the same fixed
scaling policy everywhere and never computes tile-local display statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lunarpits.tiling.ml_tiles import (  # noqa: E402
    DEFAULT_METERS_PER_PIXEL,
    DEFAULT_TILE_SIZE_M,
    MOON_RADIUS_M,
    MlScalingPolicy,
    apply_ml_normalization,
    get_ml_tile_for_latlon,
    latlon_to_global_xy,
    load_ml_policy,
    pixel_in_tile,
    render_ml_preview,
    validate_production_filename,
)


def lunar_global_eqc_crs() -> CRS:
    """Return the production first-version global spherical Moon CRS."""
    return CRS.from_proj4(
        f"+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 "
        f"+a={MOON_RADIUS_M} +b={MOON_RADIUS_M} +units=m +no_defs"
    )


def infer_product_id(tif_path: Path) -> str:
    name = tif_path.name
    for suffix in (".map.tif", ".tif"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return tif_path.stem


def default_output_path(tif_path: Path, product_id: str, tile_id: str) -> Path:
    return tif_path.with_name(f"{tile_id}_{product_id}_ml.tif")


def write_png(path: Path, arr: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path)


def crop_fixed_tile(
    tif_path: str | Path,
    *,
    lat: float,
    lon: float,
    tile_size_m: float = DEFAULT_TILE_SIZE_M,
    meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL,
    out_path: str | Path | None = None,
    product_id: str | None = None,
    policy: MlScalingPolicy | None = None,
    resampling: Resampling = Resampling.bilinear,
    preview_scale_mode: str = "source_percentile",
    preview_percentile_low: float = 1.0,
    preview_percentile_high: float = 99.0,
) -> dict[str, Any]:
    """Reproject one deterministic production ML tile from a mapped GeoTIFF."""
    tif_path = Path(tif_path)
    product_id = product_id or infer_product_id(tif_path)
    policy = policy or MlScalingPolicy()
    tile = get_ml_tile_for_latlon(lat, lon, tile_size_m=tile_size_m, meters_per_pixel=meters_per_pixel)
    target_pixel = pixel_in_tile(lat, lon, tile_size_m=tile_size_m, meters_per_pixel=meters_per_pixel)

    if out_path is None:
        out_path = default_output_path(tif_path, product_id, tile.tile_id)
    out_path = Path(out_path)
    validate_production_filename(out_path)
    preview_path = out_path.with_name(f"{out_path.stem}_preview.png")
    validate_production_filename(preview_path)
    metadata_path = out_path.with_suffix(".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dst_transform = Affine(meters_per_pixel, 0.0, tile.x_min_m, 0.0, -meters_per_pixel, tile.y_max_m)
    dst_crs = lunar_global_eqc_crs()
    raw_projected = np.full((tile.tile_size_px, tile.tile_size_px), policy.nodata_value, dtype="float32")

    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError(f"Input TIFF has no CRS: {tif_path}")

        if preview_scale_mode == "source_percentile":
            preview_min, preview_max = source_percentile_limits(
                src,
                percentile_low=preview_percentile_low,
                percentile_high=preview_percentile_high,
            )
            preview_policy = MlScalingPolicy(
                normalization_policy=policy.normalization_policy,
                preview_min=preview_min,
                preview_max=preview_max,
                ml_clip_min=policy.ml_clip_min,
                ml_clip_max=policy.ml_clip_max,
                nodata_value=policy.nodata_value,
            )
        elif preview_scale_mode == "fixed":
            preview_min, preview_max = policy.preview_min, policy.preview_max
            preview_policy = policy
        else:
            raise ValueError(f"Unknown preview_scale_mode: {preview_scale_mode}")

        src_arr = src.read(1)
        reproject(
            source=src_arr,
            destination=raw_projected,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=policy.nodata_value,
            resampling=resampling,
        )

        ml_arr = apply_ml_normalization(raw_projected, policy)
        invalid = ~np.isfinite(ml_arr)
        if invalid.any():
            ml_arr = ml_arr.copy()
            ml_arr[invalid] = policy.nodata_value

        profile = src.profile.copy()
        block_size = 256 if tile.tile_size_px >= 256 else 128
        profile.update(
            driver="GTiff",
            height=tile.tile_size_px,
            width=tile.tile_size_px,
            count=1,
            dtype="float32",
            crs=dst_crs,
            transform=dst_transform,
            nodata=policy.nodata_value,
            compress="deflate",
            tiled=True,
            blockxsize=block_size,
            blockysize=block_size,
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(ml_arr.astype("float32", copy=False), 1)

        source_profile = {
            "crs": str(src.crs),
            "transform": [src.transform.a, src.transform.b, src.transform.c, src.transform.d, src.transform.e, src.transform.f],
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
        }

    preview = render_ml_preview(ml_arr, preview_policy)
    write_png(preview_path, preview)

    valid = np.isfinite(raw_projected) & (raw_projected != policy.nodata_value)
    ml_valid = np.isfinite(ml_arr) & (ml_arr != policy.nodata_value)
    metadata = {
        "source_nac_id": product_id,
        "source_tif": str(tif_path),
        "ml_tile_path": str(out_path),
        "ml_preview_path": str(preview_path),
        "metadata_path": str(metadata_path),
        "tile_i": tile.tile_i,
        "tile_j": tile.tile_j,
        "tile_id": tile.tile_id,
        "tile_size_m": tile.tile_size_m,
        "meters_per_pixel": tile.meters_per_pixel,
        "tile_size_px": tile.tile_size_px,
        "projection": "spherical_equirectangular_moon_first_version",
        "radius": MOON_RADIUS_M,
        "latitude_type": "Planetocentric",
        "longitude_direction": "PositiveEast",
        "longitude_domain": 360,
        "center_latitude": 0.0,
        "center_longitude": 0.0,
        "x_min_m": tile.x_min_m,
        "x_max_m": tile.x_max_m,
        "y_min_m": tile.y_min_m,
        "y_max_m": tile.y_max_m,
        "ul_x": tile.ul_x,
        "ul_y": tile.ul_y,
        "target": {
            "lat": lat,
            "lon_original": lon,
            "global_x_m": target_pixel["x_m"],
            "global_y_m": target_pixel["y_m"],
            "pixel_x": target_pixel["pixel_x"],
            "pixel_y": target_pixel["pixel_y"],
        },
        "normalization_policy": policy.normalization_policy,
        "normalization_constants": {
            "ml_clip_min": policy.ml_clip_min,
            "ml_clip_max": policy.ml_clip_max,
        },
        "preview_scaling_policy": "source_product_percentile" if preview_scale_mode == "source_percentile" else "fixed_global_clip_for_preview",
        "preview_scaling_constants": {
            "preview_scale_mode": preview_scale_mode,
            "preview_percentile_low": preview_percentile_low if preview_scale_mode == "source_percentile" else None,
            "preview_percentile_high": preview_percentile_high if preview_scale_mode == "source_percentile" else None,
            "preview_min": preview_min,
            "preview_max": preview_max,
        },
        "nodata_value": policy.nodata_value,
        "valid_pixel_fraction": float(valid.mean()) if valid.size else 0.0,
        "source_value_range_before_ml_normalization": _range(raw_projected, policy.nodata_value),
        "ml_value_range": _range(ml_arr, policy.nodata_value),
        "preview_png_value_range": [int(preview.min()), int(preview.max())] if preview.size else [0, 0],
        "source_raster": source_profile,
        "output_transform": [dst_transform.a, dst_transform.b, dst_transform.c, dst_transform.d, dst_transform.e, dst_transform.f],
        "output_crs": str(dst_crs),
        "output_type": "ML tile and ML-preview only",
        "resampling": resampling.name,
        "method": "post_cam2map_global_grid_ml_tile",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def source_percentile_limits(
    src: rasterio.io.DatasetReader,
    *,
    percentile_low: float,
    percentile_high: float,
    max_pixels: int = 2_000_000,
) -> tuple[float, float]:
    """Compute grayscale limits once from the whole source NAC product."""
    if not 0.0 <= percentile_low < percentile_high <= 100.0:
        raise ValueError("Preview percentiles must satisfy 0 <= low < high <= 100.")
    scale = max(1.0, (src.width * src.height / float(max_pixels)) ** 0.5)
    out_width = max(1, int(src.width / scale))
    out_height = max(1, int(src.height / scale))
    sample = src.read(1, out_shape=(out_height, out_width), masked=True)
    values = np.asarray(sample.compressed(), dtype="float64")
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot compute source grayscale limits: source has no valid finite pixels.")
    lo, hi = np.percentile(values, [percentile_low, percentile_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"Invalid source grayscale limits: {lo}, {hi}")
    return float(lo), float(hi)


def _range(arr: np.ndarray, nodata_value: float) -> list[float | None]:
    valid = np.isfinite(arr) & (arr != nodata_value)
    if not valid.any():
        return [None, None]
    return [float(np.nanmin(arr[valid])), float(np.nanmax(arr[valid]))]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a deterministic production ML lunar tile from an already mapped LROC TIFF.")
    parser.add_argument("tif_path", help="Input map-projected LROC GeoTIFF")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--tile-size-m", type=float, default=DEFAULT_TILE_SIZE_M)
    parser.add_argument("--tile-size-km", type=float, default=None, help="Compatibility alias; overrides --tile-size-m when provided")
    parser.add_argument("--meters-per-pixel", type=float, default=DEFAULT_METERS_PER_PIXEL)
    parser.add_argument("--pixel-resolution", type=float, default=None, help="Compatibility alias for --meters-per-pixel")
    parser.add_argument("--product-id", default=None)
    parser.add_argument("--out", default=None, help="Output ML tile TIFF path")
    parser.add_argument("--policy-config", default=str(PROJECT_ROOT / "config" / "ml_tile_policy.yaml"))
    parser.add_argument("--normalization-policy", choices=["preserve_float32", "fixed_global_clip", "dataset_percentile_constants"], default=None)
    parser.add_argument("--preview-min", type=float, default=None)
    parser.add_argument("--preview-max", type=float, default=None)
    parser.add_argument("--ml-clip-min", type=float, default=None)
    parser.add_argument("--ml-clip-max", type=float, default=None)
    parser.add_argument("--nodata-value", type=float, default=None)
    parser.add_argument("--nearest", action="store_true", help="Use nearest-neighbor resampling instead of bilinear")
    parser.add_argument(
        "--preview-scale-mode",
        choices=["source_percentile", "fixed"],
        default="source_percentile",
        help="source_percentile computes grayscale limits from the whole source NAC, then applies them to the tile.",
    )
    parser.add_argument("--preview-percentile-low", type=float, default=1.0)
    parser.add_argument("--preview-percentile-high", type=float, default=99.0)
    return parser


def policy_from_args(args: argparse.Namespace) -> MlScalingPolicy:
    policy = load_ml_policy(args.policy_config)
    updates = policy.to_dict()
    for arg_name, key in (
        ("normalization_policy", "normalization_policy"),
        ("preview_min", "preview_min"),
        ("preview_max", "preview_max"),
        ("ml_clip_min", "ml_clip_min"),
        ("ml_clip_max", "ml_clip_max"),
        ("nodata_value", "nodata_value"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            updates[key] = value
    return MlScalingPolicy(**updates)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tile_size_m = args.tile_size_km * 1000.0 if args.tile_size_km is not None else args.tile_size_m
    meters_per_pixel = args.pixel_resolution if args.pixel_resolution is not None else args.meters_per_pixel
    resampling = Resampling.nearest if args.nearest else Resampling.bilinear
    metadata = crop_fixed_tile(
        args.tif_path,
        lat=args.lat,
        lon=args.lon,
        tile_size_m=tile_size_m,
        meters_per_pixel=meters_per_pixel,
        out_path=args.out,
        product_id=args.product_id,
        policy=policy_from_args(args),
        resampling=resampling,
        preview_scale_mode=args.preview_scale_mode,
        preview_percentile_low=args.preview_percentile_low,
        preview_percentile_high=args.preview_percentile_high,
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
