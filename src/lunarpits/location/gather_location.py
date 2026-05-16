"""Location-centered gatherer for LROC NAC and contextual lunar datasets."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from lunarpits.location.context_sources import (
    load_context_config,
    sample_diviner_context,
    sample_grail_context,
    sample_lola_context,
    write_json,
)
from lunarpits.location.lroc_search import find_lroc_nac_products_for_location, select_diverse_nac_observations
from lunarpits.location.lroc_search import ensure_nac_equatorial_footprints
from lunarpits.location.models import ProcessedNacProduct, TargetLocation
from lunarpits.processing.identifiers import normalize_product_id
from lunar_tile_pipeline.lroc import find_lroc_nac_edr_for_tile_from_file, select_top_lroc_nac_for_tile
from lunar_tile_pipeline.tiling import get_tile_for_latlon


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def location_output_dir_name(lat: float, lon: float) -> str:
    return f"{lat:.6f}_{lon:.6f}"


def safe_location_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return cleaned.strip("._") or "location"


def gather_location_context(
    lat: float,
    lon: float,
    radius_km: float = 5.0,
    max_nac: int = 15,
    crop_size_m: float = 1024.0,
    pixel_resolution: float = 0.5,
    tile_size_km: float = 1.0,
    process_nac: bool = True,
    keep_temp: bool = False,
    refresh_cache: bool = False,
    force: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    max_volume: int = 80,
    output_name: str | None = None,
) -> dict[str, Any]:
    target = TargetLocation(lat=lat, lon=lon, radius_km=radius_km)
    out_dir_name = safe_location_label(output_name) if output_name else location_output_dir_name(lat, lon)
    out_dir = PROJECT_ROOT / "data" / "location_context" / out_dir_name
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    tile = get_tile_for_latlon(lat, lon, tile_size_km=tile_size_km)
    write_json(out_dir / "tile_metadata.json", tile.to_metadata())

    try:
        footprints_path = ensure_nac_equatorial_footprints(refresh_cache=refresh_cache)
        df = find_lroc_nac_edr_for_tile_from_file(tile, footprints_path)
        selected = select_top_lroc_nac_for_tile(df, max_products=max_nac)
        selected_ids = set(selected["product_id"]) if not selected.empty else set()
        df["selected"] = df["product_id"].isin(selected_ids) if not df.empty else []
    except Exception as exc:
        warnings.append(f"Fixed-tile footprint search failed; falling back to location/radius search: {exc}")
        df = find_lroc_nac_products_for_location(
            lat,
            lon,
            radius_km=radius_km,
            max_products=max_nac,
            max_volume=max_volume,
            refresh_cache=refresh_cache,
            verbose=verbose,
        )
        warnings.extend(str(w) for w in df.attrs.get("warnings", [])[:20])
        selected = select_diverse_nac_observations(df, max_products=max_nac)

    products_csv = out_dir / "nac_products.csv"
    products_parquet = out_dir / "nac_products.parquet"
    df.to_csv(products_csv, index=False)
    df.to_parquet(products_parquet, index=False)

    config = load_context_config(PROJECT_ROOT / "config" / "context_sources.yaml")
    lola = sample_lola_context(lat, lon, radius_km, config.get("lola", {}))
    grail = sample_grail_context(lat, lon, radius_km, config.get("grail", {}))
    diviner = sample_diviner_context(lat, lon, radius_km, config.get("diviner", {}))
    write_json(out_dir / "lola" / "lola_context.json", lola)
    write_json(out_dir / "grail" / "grail_context.json", grail)
    write_json(out_dir / "diviner" / "diviner_context.json", diviner)

    processed_results: list[ProcessedNacProduct] = []
    if dry_run or not process_nac:
        note = "Dry run: NAC products were selected but not processed." if dry_run else "NAC processing disabled."
        warnings.append(note)
    else:
        selected_product_ids = {normalize_product_id(str(row["product_id"])) for _, row in selected.iterrows()}
        _remove_stale_location_products(crops_dir, selected_product_ids)
        for _, row in selected.iterrows():
            product_id = normalize_product_id(str(row["product_id"]))
            result = _process_and_crop_product(
                product_id,
                lat=lat,
                lon=lon,
                tile_size_km=tile_size_km,
                pixel_resolution=pixel_resolution,
                crops_dir=crops_dir,
                keep_temp=keep_temp,
                force=force,
                verbose=verbose,
                max_volume=max_volume,
            )
            processed_results.append(result)
            if result.error:
                warnings.append(f"{product_id}: {result.error}")

    context = {
        "target": target.to_dict(),
        "tile": tile.to_metadata(),
        "lroc_nac": {
            "num_found": int(len(df)),
            "num_selected": int(len(selected)),
            "products_csv": str(products_csv),
            "products_parquet": str(products_parquet),
            "selection_note": _selection_note(df),
            "processed": [item.to_dict() for item in processed_results],
        },
        "lola": lola,
        "grail": grail,
        "diviner": diviner,
        "warnings": warnings,
    }
    write_json(out_dir / "location_context.json", context)
    return context


def _remove_stale_location_products(crops_dir: Path, selected_product_ids: set[str]) -> None:
    """Keep generated coordinate-folder rasters aligned with the current selected product set."""
    if not crops_dir.exists():
        return
    for pattern in ("*.map.tif", "*.map.tif.msk", "*.quicklook.png", "*_crop.tif", "*_crop.tif.msk", "*_crop.quicklook.png"):
        for path in crops_dir.glob(pattern):
            product_id = path.name.split("_crop", 1)[0].split(".map", 1)[0].split(".quicklook", 1)[0]
            if product_id not in selected_product_ids:
                path.unlink(missing_ok=True)


def _selection_note(df: Any) -> str:
    if df.empty:
        return "No LROC NAC EDR footprint hits found for this coordinate/radius."
    if "coverage_method" in df:
        methods = ", ".join(sorted(str(v) for v in df["coverage_method"].dropna().unique()))
        return f"Selected products are confirmed by footprint geometry ({methods}); center coordinates were not used as coverage proof."
    if "incidence_angle" not in df or not df["incidence_angle"].notna().any():
        return "Selected products have footprint coverage; angle-based diversity is limited because incidence/sun metadata is unavailable."
    return "Selected products prefer known point coverage and diverse incidence/sun geometry."


def _process_and_crop_product(
    product_id: str,
    *,
    lat: float,
    lon: float,
    tile_size_km: float,
    pixel_resolution: float,
    crops_dir: Path,
    keep_temp: bool,
    force: bool,
    verbose: bool,
    max_volume: int,
) -> ProcessedNacProduct:
    crop_tif = crops_dir / f"{product_id}.map.tif"
    try:
        if force or not crop_tif.exists():
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "process_lroc_product.py"),
                product_id,
                "--tile-lat",
                str(lat),
                "--tile-lon",
                str(lon),
                "--tile-size-km",
                str(tile_size_km),
                "--pixel-resolution",
                str(pixel_resolution),
                "--max-volume",
                str(max_volume),
                "--output-dir",
                str(crops_dir),
            ]
            if not force:
                cmd.append("--skip-if-exists")
            if keep_temp:
                cmd.append("--keep-temp")
            if verbose:
                cmd.append("--verbose")
            completed = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
            if completed.returncode != 0:
                error = (completed.stderr or completed.stdout or "").strip()
                return ProcessedNacProduct(product_id=product_id, status="process_failed", error=error)

        if not crop_tif.exists():
            return ProcessedNacProduct(product_id=product_id, status="missing_output", error=f"Expected output missing: {crop_tif}")

        crop_quicklook = crops_dir / f"{product_id}.quicklook.png"
        _save_quicklook(crop_tif, crop_quicklook)
        return ProcessedNacProduct(
            product_id=product_id,
            status="fixed_tile_processed",
            context_tif=str(crop_tif),
            quicklook=str(crop_quicklook),
            crop_tif=str(crop_tif),
            crop_quicklook=str(crop_quicklook),
        )
    except Exception as exc:
        return ProcessedNacProduct(product_id=product_id, status="failed", error=str(exc))


def crop_raster_around_latlon(tif_path: Path, lat: float, lon: float, crop_size_m: float, out_path: Path) -> str | None:
    try:
        import rasterio
        from rasterio.windows import Window
    except Exception as exc:
        return f"rasterio is required for crop generation: {exc}"

    try:
        with rasterio.open(tif_path) as src:
            x, y = latlon_to_lunar_equirectangular_xy(src, lat, lon)
            row, col = src.index(x, y)
            res_x = abs(src.transform.a)
            res_y = abs(src.transform.e)
            half_w = max(1, int(round((crop_size_m / 2.0) / res_x)))
            half_h = max(1, int(round((crop_size_m / 2.0) / res_y)))
            col_off = max(0, int(col) - half_w)
            row_off = max(0, int(row) - half_h)
            width = min(src.width - col_off, half_w * 2)
            height = min(src.height - row_off, half_h * 2)
            if width <= 0 or height <= 0 or row < 0 or col < 0 or row >= src.height or col >= src.width:
                return "Target coordinate projects outside this raster; crop skipped."
            window = Window(col_off, row_off, width, height)
            profile = src.profile.copy()
            profile.update(height=int(height), width=int(width), transform=src.window_transform(window), compress="deflate")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(src.read(window=window))
            return None
    except Exception as exc:
        return f"Lat/lon crop is TODO for this raster metadata: {exc}"


def latlon_to_lunar_equirectangular_xy(src: Any, lat: float, lon: float) -> tuple[float, float]:
    params = _parse_equirectangular_params(src)
    radius = params["radius"]
    center_lon = params["central_meridian"]
    center_lat = params["standard_parallel_1"]
    delta_lon = ((lon - center_lon + 180.0) % 360.0) - 180.0
    x = radius * math.radians(delta_lon) * math.cos(math.radians(center_lat))
    y = radius * math.radians(lat)
    return x, y


def _parse_equirectangular_params(src: Any) -> dict[str, float]:
    if src.crs is None:
        raise ValueError("Raster has no CRS.")
    wkt = src.crs.to_wkt()
    if "Equirectangular" not in wkt:
        raise ValueError("Raster CRS is not lunar equirectangular.")

    def param(name: str) -> float:
        match = re_search_parameter(wkt, name)
        if match is None:
            raise ValueError(f"Missing CRS parameter: {name}")
        return match

    radius_match = re_search_spheroid_radius(wkt)
    return {
        "radius": radius_match or 1737400.0,
        "central_meridian": param("central_meridian"),
        "standard_parallel_1": param("standard_parallel_1"),
    }


def re_search_parameter(wkt: str, name: str) -> float | None:
    import re

    match = re.search(rf'PARAMETER\["{re.escape(name)}",\s*([-+]?\d+(?:\.\d+)?)\]', wkt)
    return float(match.group(1)) if match else None


def re_search_spheroid_radius(wkt: str) -> float | None:
    import re

    match = re.search(r'SPHEROID\["[^"]+",\s*([-+]?\d+(?:\.\d+)?)\s*,', wkt)
    return float(match.group(1)) if match else None


def _save_quicklook(tif_path: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio

    with rasterio.open(tif_path) as src:
        arr = src.read(1, masked=True)
    values = arr.compressed()
    if values.size:
        p2, p98 = np.percentile(values, [2, 98])
        shown = np.clip((arr.filled(p2) - p2) / max(p98 - p2, 1e-12), 0, 1)
    else:
        shown = np.zeros(arr.shape, dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, shown, cmap="gray")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gather location-centered lunar context data.")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--coords-csv", default=None, help="Batch CSV with lat,lon and optional label,radius_km,max_nac")
    parser.add_argument("--radius-km", type=float, default=5.0)
    parser.add_argument("--max-nac", type=int, default=15)
    parser.add_argument("--crop-size-m", type=float, default=1024.0)
    parser.add_argument("--tile-size-km", type=float, default=1.0, help="Deterministic fixed tile size for NAC outputs")
    parser.add_argument("--pixel-resolution", type=float, default=0.5)
    parser.add_argument("--process-nac", dest="process_nac", action="store_true", default=True)
    parser.add_argument("--no-process-nac", dest="process_nac", action="store_false")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-volume", type=int, default=80)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.coords_csv:
        contexts = gather_locations_from_csv(args)
        print(f"processed coordinate rows: {len(contexts)}")
        for context in contexts:
            print(
                f"{context['target']['lat']}, {context['target']['lon']}: "
                f"{context['lroc_nac']['num_found']} / {context['lroc_nac']['num_selected']}"
            )
        return 0

    if args.lat is None or args.lon is None:
        raise SystemExit("--lat and --lon are required unless --coords-csv is provided.")

    context = _gather_from_args(args, args.lat, args.lon)
    out_dir = PROJECT_ROOT / "data" / "location_context" / location_output_dir_name(args.lat, args.lon)
    print(f"location context: {out_dir / 'location_context.json'}")
    print(f"NAC found/selected: {context['lroc_nac']['num_found']} / {context['lroc_nac']['num_selected']}")
    return 0


def _gather_from_args(args: argparse.Namespace, lat: float, lon: float, *, radius_km: float | None = None, max_nac: int | None = None, label: str | None = None) -> dict[str, Any]:
    return gather_location_context(
        lat=lat,
        lon=lon,
        radius_km=args.radius_km if radius_km is None else radius_km,
        max_nac=args.max_nac if max_nac is None else max_nac,
        crop_size_m=args.crop_size_m,
        pixel_resolution=args.pixel_resolution,
        tile_size_km=args.tile_size_km,
        process_nac=args.process_nac,
        keep_temp=args.keep_temp,
        refresh_cache=args.refresh_cache,
        force=args.force,
        dry_run=args.dry_run,
        verbose=args.verbose,
        max_volume=args.max_volume,
        output_name=label,
    )


def gather_locations_from_csv(args: argparse.Namespace) -> list[dict[str, Any]]:
    import pandas as pd

    rows = pd.read_csv(args.coords_csv)
    required = {"lat", "lon"}
    missing = required - set(rows.columns)
    if missing:
        raise SystemExit(f"--coords-csv is missing required columns: {', '.join(sorted(missing))}")

    contexts: list[dict[str, Any]] = []
    for idx, row in rows.iterrows():
        label = str(row["label"]) if "label" in rows.columns and not pd.isna(row["label"]) else None
        radius_km = float(row["radius_km"]) if "radius_km" in rows.columns and not pd.isna(row["radius_km"]) else None
        max_nac = int(row["max_nac"]) if "max_nac" in rows.columns and not pd.isna(row["max_nac"]) else None
        context = _gather_from_args(
            args,
            float(row["lat"]),
            float(row["lon"]),
            radius_km=radius_km,
            max_nac=max_nac,
            label=label or f"row_{idx:04d}_{location_output_dir_name(float(row['lat']), float(row['lon']))}",
        )
        contexts.append(context)
    return contexts


if __name__ == "__main__":
    raise SystemExit(main())
