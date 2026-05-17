"""Render deterministic tile-system variants with one fixed grayscale policy.

This is a diagnostic script, not a production output path.  It exists so we can
compare different deterministic coordinate-to-tile conventions against the same
two NAC observations while holding grayscale constant.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
from affine import Affine
from PIL import Image, ImageDraw
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lunarpits.tiling.ml_tiles import (  # noqa: E402
    MOON_RADIUS_M,
    MlScalingPolicy,
    latlon_to_global_xy,
    load_ml_policy,
    normalize_lon_180,
    render_ml_preview,
)


@dataclass(frozen=True)
class TileVariant:
    name: str
    description: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    pixel_x: int
    pixel_y: int
    crs: CRS


def moon_eqc_crs(lat_ts: float = 0.0, lon_0: float = 0.0) -> CRS:
    return CRS.from_proj4(
        f"+proj=eqc +lat_ts={lat_ts} +lat_0=0 +lon_0={lon_0} +x_0=0 +y_0=0 "
        f"+a={MOON_RADIUS_M} +b={MOON_RADIUS_M} +units=m +no_defs"
    )


def xy_for_eqc(lat: float, lon: float, *, lat_ts: float = 0.0, lon_0: float = 0.0) -> tuple[float, float]:
    dlon = normalize_lon_180(lon - lon_0)
    return MOON_RADIUS_M * math.radians(dlon) * math.cos(math.radians(lat_ts)), MOON_RADIUS_M * math.radians(lat)


def floor_bounds(x: float, y: float, tile_size_m: float, origin_x: float = 0.0, origin_y: float = 0.0) -> tuple[float, float, float, float]:
    i = math.floor((x - origin_x) / tile_size_m)
    j = math.floor((y - origin_y) / tile_size_m)
    x_min = origin_x + i * tile_size_m
    y_min = origin_y + j * tile_size_m
    return x_min, y_min, x_min + tile_size_m, y_min + tile_size_m


def centered_bounds(x: float, y: float, tile_size_m: float, origin_x: float = 0.0, origin_y: float = 0.0) -> tuple[float, float, float, float]:
    i = math.floor((x - origin_x + tile_size_m / 2.0) / tile_size_m)
    j = math.floor((y - origin_y + tile_size_m / 2.0) / tile_size_m)
    cx = origin_x + i * tile_size_m
    cy = origin_y + j * tile_size_m
    half = tile_size_m / 2.0
    return cx - half, cy - half, cx + half, cy + half


def rounded_center_bounds(x: float, y: float, tile_size_m: float) -> tuple[float, float, float, float]:
    cx = round(x / tile_size_m) * tile_size_m
    cy = round(y / tile_size_m) * tile_size_m
    half = tile_size_m / 2.0
    return cx - half, cy - half, cx + half, cy + half


def target_centered_bounds(x: float, y: float, tile_size_m: float) -> tuple[float, float, float, float]:
    half = tile_size_m / 2.0
    return x - half, y - half, x + half, y + half


def variant_from_bounds(
    name: str,
    description: str,
    bounds: tuple[float, float, float, float],
    target_xy: tuple[float, float],
    crs: CRS,
    meters_per_pixel: float,
) -> TileVariant:
    x_min, y_min, x_max, y_max = bounds
    x, y = target_xy
    pixel_x = math.floor((x - x_min) / meters_per_pixel)
    pixel_y = math.floor((y_max - y) / meters_per_pixel)
    return TileVariant(name, description, x_min, y_min, x_max, y_max, int(pixel_x), int(pixel_y), crs)


def build_variants(lat: float, lon: float, tile_size_m: float, meters_per_pixel: float, variant_set: str = "broad") -> list[TileVariant]:
    x, y = latlon_to_global_xy(lat, lon)
    global_crs = moon_eqc_crs(lat_ts=0.0, lon_0=0.0)
    lat_ts_crs = moon_eqc_crs(lat_ts=lat, lon_0=0.0)
    x_lat_ts, y_lat_ts = xy_for_eqc(lat, lon, lat_ts=lat, lon_0=0.0)
    lon0_crs = moon_eqc_crs(lat_ts=0.0, lon_0=lon)
    x_lon0, y_lon0 = xy_for_eqc(lat, lon, lat_ts=0.0, lon_0=lon)

    if variant_set == "refined":
        quarter = tile_size_m / 4.0
        half = tile_size_m / 2.0
        definitions = [
            ("01_centered_origin", "best prior family: centered-origin global grid", centered_bounds(x, y, tile_size_m), (x, y), global_crs),
            ("02_centered_x_minus_q", "centered grid, global phase x - quarter tile", centered_bounds(x, y, tile_size_m, origin_x=-quarter), (x, y), global_crs),
            ("03_centered_x_plus_q", "centered grid, global phase x + quarter tile", centered_bounds(x, y, tile_size_m, origin_x=quarter), (x, y), global_crs),
            ("04_centered_y_minus_q", "centered grid, global phase y - quarter tile", centered_bounds(x, y, tile_size_m, origin_y=-quarter), (x, y), global_crs),
            ("05_centered_y_plus_q", "centered grid, global phase y + quarter tile", centered_bounds(x, y, tile_size_m, origin_y=quarter), (x, y), global_crs),
            ("06_centered_xy_minus_q", "centered grid, global phase x/y - quarter tile", centered_bounds(x, y, tile_size_m, origin_x=-quarter, origin_y=-quarter), (x, y), global_crs),
            ("07_centered_xy_plus_q", "centered grid, global phase x/y + quarter tile", centered_bounds(x, y, tile_size_m, origin_x=quarter, origin_y=quarter), (x, y), global_crs),
            ("08_centered_x_half", "same family as prior #6: centered grid x half phase", centered_bounds(x, y, tile_size_m, origin_x=half), (x, y), global_crs),
            ("09_target_centered", "diagnostic tile centered exactly on requested lat/lon", target_centered_bounds(x, y, tile_size_m), (x, y), global_crs),
            ("10_lat_parallel_centered", "centered grid with equirectangular lat_ts equal target latitude", centered_bounds(x_lat_ts, y_lat_ts, tile_size_m), (x_lat_ts, y_lat_ts), lat_ts_crs),
        ]
    else:
        definitions = [
            ("01_floor_global", "floor(x/tile), floor(y/tile), lon0=0 lat_ts=0", floor_bounds(x, y, tile_size_m), (x, y), global_crs),
            ("02_centered_origin", "old centered-origin global grid", centered_bounds(x, y, tile_size_m), (x, y), global_crs),
            ("03_round_center", "nearest tile center global grid", rounded_center_bounds(x, y, tile_size_m), (x, y), global_crs),
            ("04_half_shift_x", "floor grid shifted +half tile in x", floor_bounds(x, y, tile_size_m, origin_x=tile_size_m / 2.0), (x, y), global_crs),
            ("05_half_shift_y", "floor grid shifted +half tile in y", floor_bounds(x, y, tile_size_m, origin_y=tile_size_m / 2.0), (x, y), global_crs),
            ("06_half_shift_xy", "floor grid shifted +half tile in x and y", floor_bounds(x, y, tile_size_m, origin_x=tile_size_m / 2.0, origin_y=tile_size_m / 2.0), (x, y), global_crs),
            ("07_target_centered", "diagnostic tile centered exactly on requested lat/lon", target_centered_bounds(x, y, tile_size_m), (x, y), global_crs),
            ("08_lat_parallel", "floor grid with equirectangular lat_ts equal target latitude", floor_bounds(x_lat_ts, y_lat_ts, tile_size_m), (x_lat_ts, y_lat_ts), lat_ts_crs),
            ("09_lon0_target", "floor grid with central meridian equal target lon", floor_bounds(x_lon0, y_lon0, tile_size_m), (x_lon0, y_lon0), lon0_crs),
            ("10_one_based_floor", "floor grid offset by one tile from global origin", floor_bounds(x, y, tile_size_m, origin_x=-tile_size_m, origin_y=-tile_size_m), (x, y), global_crs),
        ]
    return [variant_from_bounds(name, desc, bounds, target_xy, crs, meters_per_pixel) for name, desc, bounds, target_xy, crs in definitions]


def render_variant(
    source_tif: Path,
    variant: TileVariant,
    *,
    product_id: str,
    out_dir: Path,
    meters_per_pixel: float,
    policy: MlScalingPolicy,
    resampling: Resampling,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{variant.name}_{product_id}"
    tif_path = out_dir / f"{base}.tif"
    png_path = out_dir / f"{base}.png"
    json_path = out_dir / f"{base}.json"

    tile_size_px_exact = (variant.x_max - variant.x_min) / meters_per_pixel
    if not math.isclose(tile_size_px_exact, round(tile_size_px_exact), abs_tol=1e-6):
        tile_size_px = int(math.ceil(tile_size_px_exact))
        effective_mpp = (variant.x_max - variant.x_min) / tile_size_px
    else:
        tile_size_px = int(round(tile_size_px_exact))
        effective_mpp = meters_per_pixel

    transform = Affine(effective_mpp, 0.0, variant.x_min, 0.0, -effective_mpp, variant.y_max)
    arr = np.full((tile_size_px, tile_size_px), policy.nodata_value, dtype="float32")
    with rasterio.open(source_tif) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=variant.crs,
            dst_nodata=policy.nodata_value,
            resampling=resampling,
        )
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=tile_size_px,
            width=tile_size_px,
            count=1,
            dtype="float32",
            crs=variant.crs,
            transform=transform,
            nodata=policy.nodata_value,
            compress="deflate",
        )
        with rasterio.open(tif_path, "w", **profile) as dst:
            dst.write(arr, 1)

    preview = render_ml_preview(arr, policy)
    Image.fromarray(preview, mode="L").save(png_path)
    valid = np.isfinite(arr) & (arr != policy.nodata_value)
    metadata = {
        "source_tif": str(source_tif),
        "product_id": product_id,
        "variant": asdict(variant) | {"crs": str(variant.crs)},
        "tile_size_px": tile_size_px,
        "requested_meters_per_pixel": meters_per_pixel,
        "effective_meters_per_pixel": effective_mpp,
        "normalization_policy": policy.normalization_policy,
        "preview_scaling_policy": "fixed_global_clip_for_preview",
        "preview_scaling_constants": {"preview_min": policy.preview_min, "preview_max": policy.preview_max},
        "valid_pixel_fraction": float(valid.mean()) if valid.size else 0.0,
        "value_range": [float(arr[valid].min()), float(arr[valid].max())] if valid.any() else [None, None],
        "tif_path": str(tif_path),
        "png_path": str(png_path),
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def make_contact_sheet(rows: list[dict[str, object]], out_path: Path, *, label: str) -> None:
    thumbs: list[tuple[str, Image.Image]] = []
    for row in rows:
        image = Image.open(row["png_path"]).convert("L").resize((160, 160), Image.Resampling.NEAREST).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.text((4, 4), str(row["variant"]["name"]).replace("_", " "), fill=(255, 255, 255))
        draw.text((4, 144), f"px {row['variant']['pixel_x']},{row['variant']['pixel_y']}", fill=(255, 255, 255))
        thumbs.append((str(row["product_id"]), image))

    cols = 5
    rows_count = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 160, rows_count * 160 + 24), "black")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 4), label, fill=(255, 255, 255))
    for idx, (_, image) in enumerate(thumbs):
        x = (idx % cols) * 160
        y = 24 + (idx // cols) * 160
        sheet.paste(image, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def source_display_policy(
    source_tif: Path,
    *,
    base_policy: MlScalingPolicy,
    scale_mode: str,
    percentile_low: float,
    percentile_high: float,
    sample_width: int = 1500,
) -> MlScalingPolicy:
    """Return fixed preview constants for one full NAC source.

    This is never tile-local.  Product modes compute one scale from the full
    mapped observation, then reuse it for every tile from that observation.
    """
    if scale_mode == "fixed":
        return base_policy
    with rasterio.open(source_tif) as src:
        out_width = min(sample_width, src.width)
        out_height = max(1, int(round(src.height * (out_width / src.width))))
        arr = src.read(1, out_shape=(out_height, out_width), masked=True, resampling=Resampling.average)
    values = arr.compressed().astype("float32")
    values = values[np.isfinite(values)]
    if values.size == 0:
        return base_policy
    if scale_mode == "product-minmax":
        lo = float(values.min())
        hi = float(values.max())
    elif scale_mode == "product-percentile":
        lo, hi = (float(v) for v in np.percentile(values, [percentile_low, percentile_high]))
    else:
        raise ValueError(f"Unknown scale mode: {scale_mode}")
    if not hi > lo:
        return base_policy
    payload = base_policy.to_dict()
    payload["preview_min"] = lo
    payload["preview_max"] = hi
    return MlScalingPolicy(**payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render ten deterministic tile-system variants with fixed grayscale.")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--source", action="append", required=True, help="PRODUCT_ID=path/to/map.tif")
    parser.add_argument(
        "--source-resolution",
        action="append",
        default=[],
        help="Optional PRODUCT_ID=meters_per_pixel. Use this to render each product at native NAC resolution.",
    )
    parser.add_argument("--out-dir", default="data/debug_tile_systems")
    parser.add_argument("--tile-size-m", type=float, default=256.0)
    parser.add_argument("--meters-per-pixel", type=float, default=1.0)
    parser.add_argument("--policy-config", default=str(PROJECT_ROOT / "config" / "ml_tile_policy.yaml"))
    parser.add_argument("--preview-min", type=float, default=None)
    parser.add_argument("--preview-max", type=float, default=None)
    parser.add_argument("--variant-set", choices=["broad", "refined"], default="broad")
    parser.add_argument(
        "--scale-mode",
        choices=["fixed", "product-minmax", "product-percentile"],
        default="fixed",
        help="Preview scaling source. Product modes compute one full-observation scale per NAC, never per tile.",
    )
    parser.add_argument("--percentile-low", type=float, default=1.0)
    parser.add_argument("--percentile-high", type=float, default=99.0)
    parser.add_argument("--nearest", action="store_true")
    return parser


def parse_source(text: str) -> tuple[str, Path]:
    if "=" not in text:
        path = Path(text)
        return path.name.split(".map.tif")[0], path
    product_id, path = text.split("=", 1)
    return product_id, Path(path)


def parse_source_resolutions(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit("--source-resolution must look like PRODUCT_ID=meters_per_pixel")
        product_id, value = item.split("=", 1)
        out[product_id] = float(value)
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy_payload = load_ml_policy(args.policy_config).to_dict()
    if args.preview_min is not None:
        policy_payload["preview_min"] = args.preview_min
    if args.preview_max is not None:
        policy_payload["preview_max"] = args.preview_max
    policy = MlScalingPolicy(**policy_payload)
    variants = build_variants(args.lat, args.lon, args.tile_size_m, args.meters_per_pixel, variant_set=args.variant_set)
    resampling = Resampling.nearest if args.nearest else Resampling.bilinear
    out_dir = Path(args.out_dir)
    source_resolutions = parse_source_resolutions(args.source_resolution)
    all_rows: list[dict[str, object]] = []
    for source_text in args.source:
        product_id, source_tif = parse_source(source_text)
        product_mpp = source_resolutions.get(product_id, args.meters_per_pixel)
        product_policy = source_display_policy(
            source_tif,
            base_policy=policy,
            scale_mode=args.scale_mode,
            percentile_low=args.percentile_low,
            percentile_high=args.percentile_high,
        )
        product_rows = [
            render_variant(
                source_tif,
                variant,
                product_id=product_id,
                out_dir=out_dir / product_id,
                meters_per_pixel=product_mpp,
                policy=product_policy,
                resampling=resampling,
            )
            for variant in variants
        ]
        all_rows.extend(product_rows)
        make_contact_sheet(
            product_rows,
            out_dir / f"{product_id}_ten_tile_systems.png",
            label=f"{product_id}: {args.scale_mode} gray {product_policy.preview_min:.6g}..{product_policy.preview_max:.6g}, {product_mpp:.3f} m/px",
        )
    (out_dir / "tile_system_summary.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "num_outputs": len(all_rows), "policy": policy.to_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
