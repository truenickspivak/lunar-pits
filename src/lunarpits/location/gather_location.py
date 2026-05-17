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
    sample_diviner_context_for_tile,
    sample_grail_context_for_tile,
    sample_lola_context_for_tile,
    write_json,
)
from lunarpits.location.lroc_search import find_lroc_nac_products_for_location, select_diverse_nac_observations
from lunarpits.location.lroc_search import ensure_nac_equatorial_footprints
from lunarpits.location.models import ProcessedNacProduct, TargetLocation
from lunarpits.processing.identifiers import normalize_product_id
from lunar_tile_pipeline.lroc import find_lroc_nac_edr_for_tile_from_file, select_top_lroc_nac_for_tile
from lunar_tile_pipeline.tiling import format_tile_size, get_tile_for_latlon
from lunarpits.tiling.ml_tiles import get_ml_tile_for_latlon


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def location_output_dir_name(lat: float, lon: float) -> str:
    return f"{lat:.6f}_{lon:.6f}"


def safe_location_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return cleaned.strip("._") or "location"


def tile_output_dir_name(tile_size_km: float) -> str:
    tile_size_m = float(tile_size_km) * 1000.0
    if tile_size_m < 1000.0 and abs(tile_size_m - round(tile_size_m)) < 1e-6:
        return f"tile_{int(round(tile_size_m))}m"
    return f"tile_{format_tile_size(tile_size_km)}km"


def site_tile_output_dir(site_name: str, tile_size_km: float | None = None) -> Path:
    """Return the canonical location folder.

    ``tile_size_km`` is accepted for backward compatibility, but tile size no
    longer creates an extra folder level because the deterministic tile is
    metadata for the location, not a separate site namespace.
    """
    return PROJECT_ROOT / "data" / "locations" / safe_location_label(site_name)


def gather_location_context(
    lat: float,
    lon: float,
    radius_km: float = 5.0,
    max_nac: int = 15,
    crop_size_m: float = 1024.0,
    pixel_resolution: float = 0.5,
    tile_size_km: float = 0.256,
    tile_method: str = "post-map-crop",
    process_nac: bool = True,
    download_context: bool = False,
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
    out_dir = site_tile_output_dir(out_dir_name, tile_size_km)
    nac_dir = out_dir / "nac"
    images_dir = nac_dir / "images"
    mapped_dir = nac_dir / "mapped"
    topology_dir = out_dir / "topology"
    gravity_dir = out_dir / "gravity"
    ir_dir = out_dir / "ir"
    images_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    tile = get_tile_for_latlon(lat, lon, tile_size_km=tile_size_km)
    duplicate_dir = find_existing_context_for_tile(tile.tile_id, PROJECT_ROOT / "data" / "locations", exclude=out_dir)
    if duplicate_dir and not force:
        existing = _read_json_if_exists(_context_json_path(duplicate_dir))
        context = {
            "target": target.to_dict(),
            "tile": tile.to_metadata(),
            "skipped_duplicate_tile": True,
            "existing_context_dir": str(duplicate_dir),
            "existing_location_metadata": str(_context_json_path(duplicate_dir)),
            "lroc_nac": existing.get("lroc_nac", {"num_found": 0, "num_selected": 0, "processed": []}),
            "topology": existing.get("topology", existing.get("lola", {})),
            "gravity": existing.get("gravity", existing.get("grail", {})),
            "ir": existing.get("ir", existing.get("diviner", {})),
            "warnings": [f"Skipped because tile {tile.tile_id} is already documented in {duplicate_dir}."],
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "metadata.json", context)
        write_json(out_dir / "tile.json", tile.to_metadata())
        return context

    write_json(out_dir / "tile.json", tile.to_metadata())

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

    products_csv = nac_dir / "products.csv"
    products_parquet = nac_dir / "products.parquet"
    nac_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(products_csv, index=False)
    df.to_parquet(products_parquet, index=False)

    config = load_context_config(PROJECT_ROOT / "config" / "context_sources.yaml")
    topology = sample_lola_context_for_tile(
        tile,
        config.get("lola", {}),
        cache_dir=PROJECT_ROOT / "data" / "cache" / "context_sources",
        download=download_context,
    )
    gravity = sample_grail_context_for_tile(
        tile,
        config.get("grail", {}),
        cache_dir=PROJECT_ROOT / "data" / "cache" / "context_sources",
        download=download_context,
    )
    ir = sample_diviner_context_for_tile(
        tile,
        config.get("diviner", {}),
        cache_dir=PROJECT_ROOT / "data" / "cache" / "context_sources",
        download=download_context,
    )
    write_json(topology_dir / "context.json", topology)
    write_json(gravity_dir / "context.json", gravity)
    write_json(ir_dir / "context.json", ir)

    processed_results: list[ProcessedNacProduct] = []
    if dry_run or not process_nac:
        note = "Dry run: NAC products were selected but not processed." if dry_run else "NAC processing disabled."
        warnings.append(note)
    else:
        selected_product_ids = {normalize_product_id(str(row["product_id"])) for _, row in selected.iterrows()}
        _remove_stale_location_products(images_dir, selected_product_ids)
        for _, row in selected.iterrows():
            product_id = normalize_product_id(str(row["product_id"]))
            result = _process_and_crop_product(
                product_id,
                lat=lat,
                lon=lon,
                tile_size_km=tile_size_km,
                pixel_resolution=pixel_resolution,
                images_dir=images_dir,
                mapped_dir=mapped_dir,
                tile_method=tile_method,
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
        "output_dir": str(out_dir),
        "tile": tile.to_metadata(),
        "lroc_nac": {
            "num_found": int(len(df)),
            "num_selected": int(len(selected)),
            "products_csv": str(products_csv),
            "products_parquet": str(products_parquet),
            "selection_note": _selection_note(df),
            "tile_method": tile_method,
            "processed": [item.to_dict() for item in processed_results],
        },
        "topology": topology,
        "gravity": gravity,
        "ir": ir,
        "lola": topology,
        "grail": gravity,
        "diviner": ir,
        "warnings": warnings,
    }
    write_json(out_dir / "metadata.json", context)
    return context


def _remove_stale_location_products(crops_dir: Path, selected_product_ids: set[str]) -> None:
    """Keep generated coordinate-folder rasters aligned with the current selected product set."""
    if not crops_dir.exists():
        return
    for pattern in ("*_ml.tif", "*_ml.tif.msk", "*_ml_preview.png", "*.json"):
        for path in crops_dir.glob(pattern):
            product_id = _product_id_from_ml_tile_name(path.name)
            if product_id not in selected_product_ids:
                path.unlink(missing_ok=True)


def _product_id_from_ml_tile_name(name: str) -> str:
    stem = name
    for suffix in ("_ml_preview.png", "_ml.tif.msk", "_ml.tif", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.rsplit("_", 1)[-1]


def find_existing_context_for_tile(tile_id: str, context_root: Path, *, exclude: Path | None = None) -> Path | None:
    if not context_root.exists():
        return None
    exclude_resolved = exclude.resolve() if exclude else None
    metadata_paths = list(context_root.glob("**/tile.json")) + list(context_root.glob("**/tile_metadata.json"))
    for metadata_path in sorted(metadata_paths):
        folder = metadata_path.parent.resolve()
        if exclude_resolved and folder == exclude_resolved:
            continue
        payload = _read_json_if_exists(metadata_path)
        if payload.get("tile_id") == tile_id:
            return metadata_path.parent
    return None


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _context_json_path(folder: Path) -> Path:
    metadata_json = folder / "metadata.json"
    if metadata_json.exists():
        return metadata_json
    context_json = folder / "context.json"
    if context_json.exists():
        return context_json
    return folder / "metadata.json"


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
    images_dir: Path,
    mapped_dir: Path,
    tile_method: str,
    keep_temp: bool,
    force: bool,
    verbose: bool,
    max_volume: int,
) -> ProcessedNacProduct:
    tile_size_m = float(tile_size_km) * 1000.0
    ml_tile = get_ml_tile_for_latlon(lat, lon, tile_size_m=tile_size_m, meters_per_pixel=pixel_resolution)
    crop_tif = images_dir / f"{ml_tile.tile_id}_{product_id}_ml.tif"
    crop_preview = crop_tif.with_name(f"{crop_tif.stem}_preview.png")
    try:
        if force or not crop_tif.exists():
            if tile_method == "fixed-cam2map":
                return ProcessedNacProduct(
                    product_id=product_id,
                    status="failed",
                    error="fixed-cam2map is disabled for production ML tiles; use post-map-crop so cam2map runs once per NAC observation.",
                )
            if tile_method == "post-map-crop":
                mapped_tif = mapped_dir / f"{product_id}.map.tif"
                if force or not mapped_tif.exists():
                    completed = _run_product_map_cam2map(
                        product_id,
                        pixel_resolution=pixel_resolution,
                        mapped_dir=mapped_dir,
                        keep_temp=keep_temp,
                        force=force,
                        verbose=verbose,
                        max_volume=max_volume,
                    )
                    if completed.returncode != 0:
                        error = (completed.stderr or completed.stdout or "").strip()
                        return ProcessedNacProduct(product_id=product_id, status="process_failed", error=error)
                completed = _run_post_map_tile_crop(
                    mapped_tif,
                    lat=lat,
                    lon=lon,
                    tile_size_km=tile_size_km,
                    pixel_resolution=pixel_resolution,
                    crop_tif=crop_tif,
                )
                if completed.returncode != 0:
                    error = (completed.stderr or completed.stdout or "").strip()
                    return ProcessedNacProduct(product_id=product_id, status="crop_failed", error=error)
            else:
                return ProcessedNacProduct(product_id=product_id, status="failed", error=f"Unknown tile method: {tile_method}")

        if not crop_tif.exists():
            return ProcessedNacProduct(product_id=product_id, status="missing_output", error=f"Expected output missing: {crop_tif}")

        return ProcessedNacProduct(
            product_id=product_id,
            status=f"{tile_method}_processed",
            context_tif=str(crop_tif),
            quicklook=str(crop_preview),
            crop_tif=str(crop_tif),
            crop_quicklook=str(crop_preview),
        )
    except Exception as exc:
        return ProcessedNacProduct(product_id=product_id, status="failed", error=str(exc))


def _run_fixed_tile_cam2map(
    product_id: str,
    *,
    lat: float,
    lon: float,
    tile_size_km: float,
    pixel_resolution: float,
    images_dir: Path,
    keep_temp: bool,
    force: bool,
    verbose: bool,
    max_volume: int,
) -> subprocess.CompletedProcess[str]:
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
        str(images_dir),
    ]
    _append_common_process_flags(cmd, keep_temp=keep_temp, force=force, verbose=verbose)
    return subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)


def _run_product_map_cam2map(
    product_id: str,
    *,
    pixel_resolution: float,
    mapped_dir: Path,
    keep_temp: bool,
    force: bool,
    verbose: bool,
    max_volume: int,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "process_lroc_product.py"),
        product_id,
        "--guide-pipeline",
        "--pixel-resolution",
        str(pixel_resolution),
        "--max-volume",
        str(max_volume),
        "--output-dir",
        str(mapped_dir),
    ]
    _append_common_process_flags(cmd, keep_temp=keep_temp, force=force, verbose=verbose)
    return subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)


def _run_post_map_tile_crop(
    mapped_tif: Path,
    *,
    lat: float,
    lon: float,
    tile_size_km: float,
    pixel_resolution: float,
    crop_tif: Path,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "crop_lroc_fixed_tile.py"),
        str(mapped_tif),
        "--lat",
        str(lat),
        "--lon",
        str(lon),
        "--tile-size-km",
        str(tile_size_km),
        "--pixel-resolution",
        str(pixel_resolution),
        "--out",
        str(crop_tif),
        "--preview-scale-mode",
        "source_percentile",
    ]
    return subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)


def _append_common_process_flags(cmd: list[str], *, keep_temp: bool, force: bool, verbose: bool) -> None:
    if not force:
        cmd.append("--skip-if-exists")
    if keep_temp:
        cmd.append("--keep-temp")
    if verbose:
        cmd.append("--verbose")


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gather location-centered lunar context data.")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--site", default=None, help="Optional clean site label for data/locations/<site>/...")
    parser.add_argument("--coords-csv", default=None, help="Batch CSV with lat,lon and optional label,radius_km,max_nac")
    parser.add_argument("--radius-km", type=float, default=5.0)
    parser.add_argument("--max-nac", type=int, default=15)
    parser.add_argument("--crop-size-m", type=float, default=1024.0)
    parser.add_argument("--tile-size-km", type=float, default=0.256, help="Deterministic fixed tile size for NAC outputs")
    parser.add_argument(
        "--tile-method",
        choices=["post-map-crop"],
        default="post-map-crop",
        help="How to create deterministic NAC tiles. post-map-crop maps each product once, then crops grid tiles.",
    )
    parser.add_argument("--pixel-resolution", type=float, default=0.5)
    parser.add_argument("--process-nac", dest="process_nac", action="store_true", default=True)
    parser.add_argument("--no-process-nac", dest="process_nac", action="store_false")
    parser.add_argument("--download-context", action="store_true", help="Download/cache configured coarse LOLA/GRAIL/Diviner context sources")
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
    site_name = args.site or location_output_dir_name(args.lat, args.lon)
    out_dir = Path(context.get("output_dir") or site_tile_output_dir(site_name, args.tile_size_km))
    print(f"location context: {out_dir / 'metadata.json'}")
    if context.get("skipped_duplicate_tile"):
        print(f"skipped duplicate tile: {context.get('existing_context_dir')}")
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
        tile_method=args.tile_method,
        process_nac=args.process_nac,
        download_context=args.download_context,
        keep_temp=args.keep_temp,
        refresh_cache=args.refresh_cache,
        force=args.force,
        dry_run=args.dry_run,
        verbose=args.verbose,
        max_volume=args.max_volume,
        output_name=label or args.site,
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
