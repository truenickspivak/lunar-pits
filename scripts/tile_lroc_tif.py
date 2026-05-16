"""Tile processed geospatial LROC NAC GeoTIFFs for ML workflows."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, bounds as window_bounds


CONTRAST_SCALE = 1000.0
STD_SCALE = 300.0
DARK_FRACTION_SCALE = 0.10
EDGE_DENSITY_SCALE = 0.05

CONTRAST_WEIGHT = 0.35
STD_WEIGHT = 0.25
EDGE_WEIGHT = 0.25
DARK_WEIGHT = 0.15


def infer_product_id(tif_path: str | Path) -> str:
    name = Path(tif_path).name
    if name.endswith(".map.tif"):
        return name.removesuffix(".map.tif")
    if name.endswith(".tif"):
        return name.removesuffix(".tif")
    return Path(name).stem


def compute_tile_stats(arr: np.ndarray, nodata: float | int | None = None) -> dict[str, float]:
    """Compute cheap conservative triage metrics for one tile."""
    values = np.asarray(arr)
    valid_mask = np.isfinite(values)
    if nodata is not None:
        valid_mask &= values != nodata

    valid_fraction = float(valid_mask.mean()) if valid_mask.size else 0.0
    if valid_fraction == 0.0:
        return {
            "valid_pixel_fraction": 0.0,
            "contrast_p98_p2": 0.0,
            "pixel_std": 0.0,
            "dark_fraction": 0.0,
            "edge_density": 0.0,
            "interest_score": 0.0,
        }

    valid_pixels = values[valid_mask].astype(np.float64, copy=False)
    p2, p5, p50, _p95, p98 = np.percentile(valid_pixels, [2, 5, 50, 95, 98])
    contrast_p98_p2 = float(p98 - p2)
    pixel_std = float(np.std(valid_pixels))
    dark_fraction = float(np.mean(valid_pixels <= p5))

    filled = values.astype(np.float64, copy=False).copy()
    filled[~valid_mask] = p50
    gy, gx = np.gradient(filled)
    grad = np.sqrt((gx * gx) + (gy * gy))
    valid_grad = grad[valid_mask]
    if valid_grad.size == 0:
        edge_density = 0.0
    else:
        edge_threshold = np.percentile(valid_grad, 95)
        edge_density = float(np.mean(valid_grad > edge_threshold))

    contrast_score = min(contrast_p98_p2 / CONTRAST_SCALE, 1.0)
    std_score = min(pixel_std / STD_SCALE, 1.0)
    dark_score = min(dark_fraction / DARK_FRACTION_SCALE, 1.0)
    edge_score = min(edge_density / EDGE_DENSITY_SCALE, 1.0)
    interest_score = (
        CONTRAST_WEIGHT * contrast_score
        + STD_WEIGHT * std_score
        + EDGE_WEIGHT * edge_score
        + DARK_WEIGHT * dark_score
    )

    return {
        "valid_pixel_fraction": valid_fraction,
        "contrast_p98_p2": contrast_p98_p2,
        "pixel_std": pixel_std,
        "dark_fraction": dark_fraction,
        "edge_density": edge_density,
        "interest_score": float(interest_score),
    }


def _iter_full_windows(width: int, height: int, tile_w: int, tile_h: int, stride_x: int, stride_y: int):
    if width < tile_w or height < tile_h:
        return
    for row_start in range(0, height - tile_h + 1, stride_y):
        for col_start in range(0, width - tile_w + 1, stride_x):
            yield Window(col_start, row_start, tile_w, tile_h)


def _tile_profile(src: rasterio.io.DatasetReader, window: Window) -> dict[str, Any]:
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        height=int(window.height),
        width=int(window.width),
        count=1,
        transform=src.window_transform(window),
        compress="deflate",
    )
    return profile


def tile_tif(
    tif_path,
    out_dir="data/tiles/nac",
    product_id=None,
    tile_size_m=512,
    stride_m=256,
    min_valid_fraction=0.20,
    interest_threshold=0.10,
    boring_keep_fraction=0.10,
    mode="training",
    seed=42,
):
    """Cut a geospatial LROC NAC GeoTIFF into fixed physical-size tiles."""
    if mode not in {"training", "inference"}:
        raise ValueError("mode must be 'training' or 'inference'.")
    if not 0.0 <= boring_keep_fraction <= 1.0:
        raise ValueError("boring_keep_fraction must be between 0 and 1.")

    tif_path = Path(tif_path)
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "nac_tiles.parquet"
    product_id = product_id or infer_product_id(tif_path)
    rng = random.Random(seed)

    rows: list[dict[str, Any]] = []
    counters = {
        "invalid_skipped": 0,
        "boring_skipped": 0,
        "boring_sampled": 0,
        "interesting_kept": 0,
    }

    with rasterio.open(tif_path) as src:
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)
        if res_x <= 0 or res_y <= 0:
            raise ValueError(f"Input raster has invalid meter-scale transform: {src.transform}")

        tile_w = max(1, int(round(tile_size_m / res_x)))
        tile_h = max(1, int(round(tile_size_m / res_y)))
        stride_x = max(1, int(round(stride_m / res_x)))
        stride_y = max(1, int(round(stride_m / res_y)))

        print(f"input path: {tif_path}")
        print(f"width/height: {src.width} / {src.height}")
        print(f"CRS: {src.crs}")
        print(f"transform: {src.transform}")
        print(f"meters per pixel: x={res_x}, y={res_y}")
        print(f"tile size in pixels: width={tile_w}, height={tile_h}")
        print(f"stride in pixels: x={stride_x}, y={stride_y}")

        for idx, window in enumerate(_iter_full_windows(src.width, src.height, tile_w, tile_h, stride_x, stride_y)):
            row_start = int(window.row_off)
            col_start = int(window.col_off)
            row_stop = row_start + int(window.height)
            col_stop = col_start + int(window.width)
            tile_id = f"{product_id}_tile_{idx:06d}"
            arr = src.read(1, window=window, masked=False)
            stats = compute_tile_stats(arr, nodata=src.nodata)

            saved = False
            tile_path: Path | None = None
            if stats["valid_pixel_fraction"] < min_valid_fraction:
                status = "invalid_skipped"
            elif stats["interest_score"] < interest_threshold:
                if mode == "training" and rng.random() < boring_keep_fraction:
                    status = "boring_sampled"
                    saved = True
                else:
                    status = "boring_skipped"
            else:
                status = "interesting_kept"
                saved = True

            counters[status] += 1

            if saved:
                tile_path = images_dir / f"{tile_id}.tif"
                with rasterio.open(tile_path, "w", **_tile_profile(src, window)) as dst:
                    dst.write(arr, 1)

            b = window_bounds(window, src.transform)
            rows.append(
                {
                    "tile_id": tile_id,
                    "product_id": product_id,
                    "source_tif": str(tif_path),
                    "tile_path": str(tile_path) if tile_path is not None else "",
                    "saved": saved,
                    "status": status,
                    "row_start": row_start,
                    "row_stop": row_stop,
                    "col_start": col_start,
                    "col_stop": col_stop,
                    "width_px": int(window.width),
                    "height_px": int(window.height),
                    "tile_size_m": float(tile_size_m),
                    "stride_m": float(stride_m),
                    "meters_per_pixel_x": float(res_x),
                    "meters_per_pixel_y": float(res_y),
                    "bounds_left": float(b[0]),
                    "bounds_bottom": float(b[1]),
                    "bounds_right": float(b[2]),
                    "bounds_top": float(b[3]),
                    "center_x": float((b[0] + b[2]) / 2.0),
                    "center_y": float((b[1] + b[3]) / 2.0),
                    **stats,
                }
            )

    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)

    total_saved = int(df["saved"].sum()) if not df.empty else 0
    print(f"number of tiles considered: {len(df)}")
    print(f"number invalid skipped: {counters['invalid_skipped']}")
    print(f"number boring skipped: {counters['boring_skipped']}")
    print(f"number boring sampled: {counters['boring_sampled']}")
    print(f"number interesting kept: {counters['interesting_kept']}")
    print(f"total saved: {total_saved}")
    print(f"path to parquet: {parquet_path}")

    return df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tile one processed geospatial LROC NAC .map.tif.")
    parser.add_argument("tif_path", help="Path to input .map.tif")
    parser.add_argument("--out-dir", default="data/tiles/nac", help="Output directory")
    parser.add_argument("--product-id", default=None, help="Optional product ID override")
    parser.add_argument("--tile-size-m", type=float, default=512, help="Physical tile size in meters")
    parser.add_argument("--stride-m", type=float, default=256, help="Physical stride in meters")
    parser.add_argument("--min-valid-fraction", type=float, default=0.20)
    parser.add_argument("--interest-threshold", type=float, default=0.10)
    parser.add_argument("--boring-keep-fraction", type=float, default=0.10)
    parser.add_argument("--mode", choices=["training", "inference"], default="training")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    tile_tif(
        args.tif_path,
        out_dir=args.out_dir,
        product_id=args.product_id,
        tile_size_m=args.tile_size_m,
        stride_m=args.stride_m,
        min_valid_fraction=args.min_valid_fraction,
        interest_threshold=args.interest_threshold,
        boring_keep_fraction=args.boring_keep_fraction,
        mode=args.mode,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

