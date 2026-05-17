"""Location-centered gatherer for LROC NAC and contextual lunar datasets."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon

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
from lunar_tile_pipeline.lroc import find_lroc_nac_edr_for_tile_from_file, find_lroc_nac_edr_for_tile_from_ode, select_top_lroc_nac_for_tile
from lunar_tile_pipeline.tiling import format_tile_size
from lunarpits.tiling.ml_tiles import MlTileSpec, get_ml_tile_for_latlon, global_xy_to_latlon, pixel_in_tile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MIN_NAC_CANDIDATE_ATTEMPTS = 25
MAX_NAC_CANDIDATE_ATTEMPTS = 50


@dataclass(frozen=True)
class LocationSearchTile:
    """Footprint-search compatible view of the production ML tile grid."""

    tile_id: str
    tile_size_km: float
    tile_size_m: float
    tile_x: int
    tile_y: int
    center_x: float
    center_y: float
    center_lat: float
    center_lon_180: float
    center_lon_360: float
    bounds_xy: tuple[float, float, float, float]
    bounds_latlon: dict[str, object]
    polygon_xy: Polygon
    polygon_latlon: Polygon
    ml_tile: MlTileSpec

    def to_metadata(self) -> dict[str, object]:
        x_min, y_min, x_max, y_max = self.bounds_xy
        return {
            "tile_id": self.tile_id,
            "tile_size_km": self.tile_size_km,
            "tile_size_m": self.tile_size_m,
            "tile_x": self.tile_x,
            "tile_y": self.tile_y,
            "tile_i": self.tile_x,
            "tile_j": self.tile_y,
            "projection": "spherical_equirectangular_moon_ml_grid",
            "moon_radius_m": 1737400.0,
            "center": {
                "x_m": self.center_x,
                "y_m": self.center_y,
                "lat": self.center_lat,
                "lon_180": self.center_lon_180,
                "lon_360": self.center_lon_360,
            },
            "bounds_xy_m": {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
            },
            "bounds_latlon": self.bounds_latlon,
            "ml_tile": self.ml_tile.to_dict(),
        }


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


def production_tile_for_latlon(lat: float, lon: float, tile_size_km: float, meters_per_pixel: float) -> LocationSearchTile:
    """Return the production ML tile with footprint-search geometry attached."""
    tile_size_m = float(tile_size_km) * 1000.0
    ml_tile = get_ml_tile_for_latlon(lat, lon, tile_size_m=tile_size_m, meters_per_pixel=meters_per_pixel)
    x_min = ml_tile.x_min_m
    x_max = ml_tile.x_max_m
    y_min = ml_tile.y_min_m
    y_max = ml_tile.y_max_m
    xy_corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)]
    lonlat_corners = []
    corner_latlon = []
    for x, y in xy_corners:
        corner_lat, corner_lon_180 = global_xy_to_latlon(x, y)
        lonlat_corners.append((corner_lon_180, corner_lat))
        corner_latlon.append(
            {
                "lat": corner_lat,
                "lon_180": corner_lon_180,
                "lon_360": ((corner_lon_180 % 360.0) + 360.0) % 360.0,
            }
        )
    lats = [float(item["lat"]) for item in corner_latlon]
    lon180s = [float(item["lon_180"]) for item in corner_latlon]
    lon360s = [float(item["lon_360"]) for item in corner_latlon]
    crosses_lon_360_wrap = (max(lon360s) - min(lon360s)) > 180.0
    bounds_latlon = {
        "corner_latlon": corner_latlon,
        "min_lat_approx": min(lats),
        "max_lat_approx": max(lats),
        "min_lon_180_approx": min(lon180s),
        "max_lon_180_approx": max(lon180s),
        "min_lon_360_approx": max(lon360s) if crosses_lon_360_wrap else min(lon360s),
        "max_lon_360_approx": min(lon360s) if crosses_lon_360_wrap else max(lon360s),
        "crosses_lon_360_wrap": crosses_lon_360_wrap,
    }
    return LocationSearchTile(
        tile_id=ml_tile.tile_id,
        tile_size_km=float(tile_size_km),
        tile_size_m=tile_size_m,
        tile_x=ml_tile.tile_i,
        tile_y=ml_tile.tile_j,
        center_x=ml_tile.center_x_m,
        center_y=ml_tile.center_y_m,
        center_lat=ml_tile.center_lat,
        center_lon_180=ml_tile.center_lon_180,
        center_lon_360=ml_tile.center_lon_360,
        bounds_xy=(x_min, y_min, x_max, y_max),
        bounds_latlon=bounds_latlon,
        polygon_xy=Polygon(xy_corners),
        polygon_latlon=Polygon(lonlat_corners),
        ml_tile=ml_tile,
    )


def gather_location_context(
    lat: float,
    lon: float,
    radius_km: float = 5.0,
    max_nac: int = 15,
    crop_size_m: float = 1024.0,
    pixel_resolution: float = 1.0,
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
    preferred_product_ids: list[str] | None = None,
) -> dict[str, Any]:
    target = TargetLocation(lat=lat, lon=lon, radius_km=radius_km)
    tile_size_m = float(tile_size_km) * 1000.0
    out_dir_name = safe_location_label(output_name) if output_name else location_output_dir_name(lat, lon)
    out_dir = site_tile_output_dir(out_dir_name, tile_size_km)
    nac_dir = out_dir / "nac"
    images_dir = nac_dir / "images"
    mapped_cache_root = PROJECT_ROOT / "data" / "cache" / "lroc_mapped"

    warnings: list[str] = []
    tile = production_tile_for_latlon(lat, lon, tile_size_km=tile_size_km, meters_per_pixel=pixel_resolution)
    duplicate_dir = find_existing_context_for_tile(tile.tile_id, PROJECT_ROOT / "data" / "locations", exclude=out_dir)
    if duplicate_dir and not force:
        existing = _read_json_if_exists(_context_json_path(duplicate_dir))
        _remove_empty_duplicate_shell(out_dir)
        context = {
            "target": target.to_dict(),
            "output_dir": str(duplicate_dir),
            "requested_output_dir": str(out_dir),
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
        return context

    images_dir.mkdir(parents=True, exist_ok=True)
    _remove_legacy_context_dirs(out_dir)
    _write_or_validate_coordinate_lock(out_dir, lat=lat, lon=lon, tile=tile, force=force)
    write_json(out_dir / "tile.json", tile.to_metadata())

    try:
        df = find_lroc_nac_edr_for_tile_from_ode(tile, limit=max(max_nac * 10, MAX_NAC_CANDIDATE_ATTEMPTS))
        if df.empty:
            footprints_path = ensure_nac_equatorial_footprints(refresh_cache=refresh_cache)
            warnings.append("ODE EDRNAC4 footprint search returned no products; falling back to cached LROC equatorial science shapefile.")
            df = find_lroc_nac_edr_for_tile_from_file(tile, footprints_path)
        candidate_limit = _candidate_attempt_limit(len(df), max_nac)
        selected = select_top_lroc_nac_for_tile(df, max_products=candidate_limit)
        selected = _prioritize_preferred_products(selected, preferred_product_ids, max_products=candidate_limit)
        candidate_ids = set(selected["product_id"]) if not selected.empty else set()
        preview_ids = set(selected.head(max_nac)["product_id"]) if not selected.empty else set()
        df["processing_candidate"] = False if df.empty else df["product_id"].isin(candidate_ids)
        df["selected"] = False if df.empty else df["product_id"].isin(preview_ids)
        df["catalog_recommended"] = False if df.empty else df["product_id"].isin(_preferred_product_set(preferred_product_ids))
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
        candidate_limit = _candidate_attempt_limit(len(df), max_nac)
        selected = select_diverse_nac_observations(df, max_products=candidate_limit)
        selected = _prioritize_preferred_products(selected, preferred_product_ids, max_products=candidate_limit)
        candidate_ids = set(selected["product_id"]) if not selected.empty and "product_id" in selected else set()
        preview_ids = set(selected.head(max_nac)["product_id"]) if not selected.empty and "product_id" in selected else set()
        if "product_id" in df:
            df["processing_candidate"] = df["product_id"].isin(candidate_ids)
            df["selected"] = df["product_id"].isin(preview_ids)
            df["catalog_recommended"] = df["product_id"].isin(_preferred_product_set(preferred_product_ids))

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
    processed_results: list[ProcessedNacProduct] = []
    if dry_run or not process_nac:
        note = "Dry run: NAC products were selected but not processed." if dry_run else "NAC processing disabled."
        warnings.append(note)
    else:
        candidate_product_ids = {normalize_product_id(str(row["product_id"])) for _, row in selected.iterrows()}
        _remove_stale_location_products(images_dir, candidate_product_ids)
        for _, row in selected.iterrows():
            if len([item for item in processed_results if not item.error and item.context_tif]) >= max_nac:
                break
            product_id = normalize_product_id(str(row["product_id"]))
            product_pixel_resolution = _effective_product_pixel_resolution(
                row,
                default_pixel_resolution=pixel_resolution,
                tile_size_m=tile_size_m,
            )
            result = _process_and_crop_product(
                product_id,
                lat=lat,
                lon=lon,
                tile_size_km=tile_size_km,
                pixel_resolution=product_pixel_resolution,
                img_url=str(row.get("img_url") or "") or None,
                images_dir=images_dir,
                mapped_dir=_mapped_cache_dir(mapped_cache_root, product_pixel_resolution),
                tile_method=tile_method,
                keep_temp=keep_temp,
                force=force,
                verbose=verbose,
                max_volume=max_volume,
            )
            processed_results.append(result)
            if result.error:
                warnings.append(f"{product_id}: {result.error}")
        accepted_ids = {item.product_id for item in processed_results if not item.error and item.context_tif}
        attempted_ids = {item.product_id for item in processed_results}
        if "product_id" in df:
            df["processing_attempted"] = df["product_id"].isin(attempted_ids)
            df["selected"] = df["product_id"].isin(accepted_ids)
            df.to_csv(products_csv, index=False)
            df.to_parquet(products_parquet, index=False)

    context = {
        "target": target.to_dict(),
        "output_dir": str(out_dir),
        "tile": tile.to_metadata(),
        "lroc_nac": {
            "num_found": int(len(df)),
            "num_selected": int(len([item for item in processed_results if not item.error and item.context_tif])) if not dry_run and process_nac else int(min(len(selected), max_nac)),
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
    write_json(out_dir / "tile.json", _tile_training_payload(context))
    write_json(out_dir / "audit.json", context)
    (out_dir / "metadata.json").unlink(missing_ok=True)
    _write_training_manifests(out_dir, context)
    return context


def _tile_training_payload(context: dict[str, Any]) -> dict[str, Any]:
    """Build the compact training/inspection unit for a location folder.

    ``audit.json`` keeps verbose pipeline provenance. ``tile.json`` is the file
    the CNN loader and a human reviewer should be able to read without knowing
    the history of the pipeline internals.
    """
    tile = context.get("tile", {})
    target = context.get("target", {})
    lroc_nac = context.get("lroc_nac", {})
    out_dir = Path(str(context.get("output_dir", ".")))
    return {
        "schema_version": "location_tile_v2",
        "location_id": out_dir.name if str(out_dir) != "." else "",
        "target": {
            "lat": target.get("lat"),
            "lon_180": target.get("lon"),
            "lon_360": _normalize_lon_360(target.get("lon")),
            "radius_km": target.get("radius_km"),
        },
        "tile": _compact_tile_payload(tile),
        "label": context.get("dataset_label", {"split_group": "unlabeled", "label": "unlabeled"}),
        "images": _compact_image_entries(lroc_nac.get("processed", []), out_dir),
        "context": {
            "topology": _compact_context_source(context.get("topology", context.get("lola", {}))),
            "gravity": _compact_context_source(context.get("gravity", context.get("grail", {}))),
            "ir": _compact_ir_context(context.get("ir", context.get("diviner", {}))),
        },
        "annotations": {
            "annotations_csv": "annotations.csv",
            "masks_dir": "nac/masks",
            "mask_status": "needs_manual_mask" if context.get("dataset_label", {}).get("split_group") == "positive" else "not_required",
        },
        "selection": {
            "num_found": lroc_nac.get("num_found"),
            "num_selected": lroc_nac.get("num_selected"),
            "products_csv": "nac/products.csv",
            "tile_method": lroc_nac.get("tile_method"),
            "note": lroc_nac.get("selection_note"),
        },
        "policy": {
            "image_and_preview_are_same_training_policy": True,
            "preview_policy": "fixed_global_scaling",
            "enhanced_browse_images": False,
        },
        "warnings": context.get("warnings", []),
    }


def _compact_tile_payload(tile: dict[str, Any]) -> dict[str, Any]:
    center = tile.get("center", {})
    bounds = tile.get("bounds_xy_m", {})
    latlon = tile.get("bounds_latlon", {})
    ml_tile = tile.get("ml_tile", {})
    return {
        "tile_id": tile.get("tile_id"),
        "tile_i": tile.get("tile_i", tile.get("tile_x")),
        "tile_j": tile.get("tile_j", tile.get("tile_y")),
        "tile_size_m": tile.get("tile_size_m"),
        "meters_per_pixel": ml_tile.get("meters_per_pixel"),
        "width_px": ml_tile.get("tile_size_px"),
        "height_px": ml_tile.get("tile_size_px"),
        "center": {
            "lat": center.get("lat"),
            "lon_180": center.get("lon_180"),
            "lon_360": center.get("lon_360"),
            "x_m": center.get("x_m"),
            "y_m": center.get("y_m"),
        },
        "bounds_xy_m": bounds,
        "bounds_latlon": {
            "min_lat": latlon.get("min_lat_approx"),
            "max_lat": latlon.get("max_lat_approx"),
            "min_lon_180": latlon.get("min_lon_180_approx"),
            "max_lon_180": latlon.get("max_lon_180_approx"),
            "min_lon_360": latlon.get("min_lon_360_approx"),
            "max_lon_360": latlon.get("max_lon_360_approx"),
        },
    }


def _compact_image_entries(processed: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for rank, item in enumerate([p for p in processed if not p.get("error")], start=1):
        image_value = str(item.get("context_tif") or item.get("crop_tif") or "")
        preview_value = str(item.get("quicklook") or item.get("crop_quicklook") or "")
        image_path = Path(image_value) if image_value else None
        preview_path = Path(preview_value) if preview_value else None
        sidecar_path = image_path.with_suffix(".json") if image_path else None
        sidecar = _read_json_if_exists(sidecar_path)
        images.append(
            {
                "rank": rank,
                "product_id": item.get("product_id"),
                "image_tif": _relative_or_string(image_path, out_dir),
                "preview_png": _relative_or_string(preview_path, out_dir),
                "sidecar_json": _relative_or_string(sidecar_path, out_dir),
                "valid_pixel_fraction": sidecar.get("valid_pixel_fraction"),
                "resolution_m_per_pixel": sidecar.get("meters_per_pixel"),
                "incidence_angle": sidecar.get("incidence_angle"),
                "emission_angle": sidecar.get("emission_angle"),
                "phase_angle": sidecar.get("phase_angle"),
            }
        )
    return images


def _compact_context_source(payload: dict[str, Any]) -> dict[str, Any]:
    stats = payload.get("stats", {})
    compact: dict[str, Any] = {
        "available": bool(payload.get("available")),
        "dataset": payload.get("dataset"),
        "sample_mode": payload.get("sample_mode"),
        "native_resolution_m_per_pixel_estimate": payload.get("native_resolution_m_per_pixel_estimate"),
        "tile_pixels_across_native_estimate": payload.get("tile_pixels_across_native_estimate"),
        "stats": _compact_stats(stats),
    }
    for key in ("center_value", "center_value_m", "nearest_lat", "nearest_lon_360", "nearest_distance_degrees_approx", "reason", "note"):
        if key in payload:
            compact[key] = payload.get(key)
    return {key: value for key, value in compact.items() if value is not None and value != {}}


def _compact_ir_context(payload: dict[str, Any]) -> dict[str, Any]:
    compact = _compact_context_source(payload)
    rasters: dict[str, Any] = {}
    for name, raster in (payload.get("rasters") or {}).items():
        rasters[name] = _compact_context_source(raster)
    if rasters:
        compact["rasters"] = rasters
    return compact


def _compact_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: stats.get(key)
        for key in ("min", "max", "mean", "median", "std", "valid_pixel_fraction")
        if key in stats
    }


def _relative_or_string(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path)


def _normalize_lon_360(value: Any) -> float | None:
    try:
        lon = float(value)
    except (TypeError, ValueError):
        return None
    return ((lon % 360.0) + 360.0) % 360.0


def _candidate_attempt_limit(num_found: int, max_nac: int) -> int:
    """Return how many ranked products to try before admitting a short stack.

    A footprint hit is only a candidate. ISIS processing, final tile coverage,
    and artifact QA can still reject it, so the gatherer must walk deeper than
    the first five ranked products. The cap keeps accidental broad queries from
    attempting hundreds of long cam2map jobs for one coordinate.
    """
    if num_found <= 0 or max_nac <= 0:
        return 0
    desired = max(max_nac * 5, MIN_NAC_CANDIDATE_ATTEMPTS)
    return min(int(num_found), min(desired, MAX_NAC_CANDIDATE_ATTEMPTS))


def _remove_legacy_context_dirs(out_dir: Path) -> None:
    """Remove old per-source context folders; context now lives in tile.json."""
    for name in ("topology", "gravity", "ir", "lola", "grail", "diviner"):
        path = out_dir / name
        if path.exists() and path.is_dir():
            shutil.rmtree(path)


def _remove_empty_duplicate_shell(out_dir: Path) -> None:
    """Remove the unused requested folder when its tile is already documented."""
    if not out_dir.exists():
        return
    try:
        has_files = any(path.is_file() for path in out_dir.rglob("*"))
    except OSError:
        return
    if not has_files:
        shutil.rmtree(out_dir, ignore_errors=True)


def _write_training_manifests(out_dir: Path, context: dict[str, Any]) -> None:
    """Write the same training package files for ad-hoc and dataset runs."""
    import pandas as pd

    target = context.get("target", {})
    location_id = out_dir.name
    label = context.get("dataset_label", {})
    split_group = label.get("split_group", "unlabeled")
    class_label = label.get("label", "unlabeled")
    annotations = {
        "location_id": location_id,
        "split_group": split_group,
        "label": class_label,
        "label_source": label.get("label_source", "manual_or_ad_hoc_coordinate"),
        "name": label.get("name", location_id),
        "lat": target.get("lat"),
        "lon_180": target.get("lon"),
        "lon_360": target.get("lon"),
        "mask_status": "needs_manual_mask" if split_group == "positive" else "not_required",
        "notes": "",
    }
    pd.DataFrame([annotations]).to_csv(out_dir / "annotations.csv", index=False)

    masks_dir = out_dir / "nac" / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    readme = masks_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "Instance masks for positive skylight/lava-tube examples go here. "
            "Use the corresponding NAC image tile as the coordinate frame.\n",
            encoding="utf-8",
        )

    processed = context.get("lroc_nac", {}).get("processed", [])
    rows: list[dict[str, Any]] = []
    for product in processed:
        product_id = product.get("product_id", "")
        sidecar = out_dir / "nac" / "images" / f"{product_id}.json"
        sidecar_payload = _read_json_if_exists(sidecar) if sidecar.exists() else {}
        valid_fraction = None
        if sidecar_payload:
            valid_fraction = sidecar_payload.get("valid_pixel_fraction")
        rows.append(
            {
                "location_id": location_id,
                "split_group": split_group,
                "label": class_label,
                "lat": target.get("lat"),
                "lon": target.get("lon"),
                "tile_id": context.get("tile", {}).get("tile_id"),
                "product_id": product_id,
                "image_path": product.get("context_tif") or product.get("crop_tif") or "",
                "preview_path": product.get("quicklook") or product.get("crop_quicklook") or "",
                "metadata_path": str(sidecar) if sidecar.exists() else "",
                "mapped_tif": str(sidecar_payload.get("source_tif") or ""),
                "valid_pixel_fraction": valid_fraction,
                "tile_json": str(out_dir / "tile.json"),
                "annotations_csv": str(out_dir / "annotations.csv"),
                "mask_status": annotations["mask_status"],
                "gravity_available": bool(context.get("gravity", {}).get("available")),
                "topology_available": bool(context.get("topology", {}).get("available")),
                "ir_available": bool(context.get("ir", {}).get("available")),
            }
        )
    if not rows:
        rows.append(
            {
                "location_id": location_id,
                "split_group": split_group,
                "label": class_label,
                "lat": target.get("lat"),
                "lon": target.get("lon"),
                "tile_id": context.get("tile", {}).get("tile_id"),
                "product_id": "",
                "image_path": "",
                "preview_path": "",
                "metadata_path": "",
                "mapped_tif": "",
                "valid_pixel_fraction": None,
                "tile_json": str(out_dir / "tile.json"),
                "annotations_csv": str(out_dir / "annotations.csv"),
                "mask_status": annotations["mask_status"],
                "gravity_available": bool(context.get("gravity", {}).get("available")),
                "topology_available": bool(context.get("topology", {}).get("available")),
                "ir_available": bool(context.get("ir", {}).get("available")),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    try:
        manifest.to_parquet(out_dir / "manifest.parquet", index=False)
    except Exception:
        pass


def _remove_stale_location_products(crops_dir: Path, selected_product_ids: set[str]) -> None:
    """Keep generated coordinate-folder rasters aligned with the current selected product set."""
    if not crops_dir.exists():
        return
    for pattern in ("*_ml.tif", "*_ml.tif.msk", "*_ml_preview.png", "*.json"):
        for path in crops_dir.glob(pattern):
            product_id = _product_id_from_ml_tile_name(path.name)
            if product_id not in selected_product_ids:
                path.unlink(missing_ok=True)
    for pattern in ("*.tif", "*.tif.msk", "*.png", "*.json"):
        for path in crops_dir.glob(pattern):
            product_id = path.stem
            if product_id.endswith("_preview"):
                product_id = product_id[: -len("_preview")]
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
        payload_tile_id = payload.get("tile_id") or payload.get("tile", {}).get("tile_id")
        if payload_tile_id == tile_id:
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
    audit_json = folder / "audit.json"
    if audit_json.exists():
        return audit_json
    tile_json = folder / "tile.json"
    if tile_json.exists():
        return tile_json
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
    if "catalog_recommended" in df and bool(df["catalog_recommended"].fillna(False).any()):
        return (
            "Selected products are footprint-confirmed hits. Official LROC pit-catalog image IDs were boosted when "
            "they also intersected the deterministic tile; center coordinates were not used as coverage proof."
        )
    if "coverage_method" in df:
        methods = ", ".join(sorted(str(v) for v in df["coverage_method"].dropna().unique()))
        return f"Selected products are confirmed by footprint geometry ({methods}); center coordinates were not used as coverage proof."
    if "incidence_angle" not in df or not df["incidence_angle"].notna().any():
        return "Selected products have footprint coverage; angle-based diversity is limited because incidence/sun metadata is unavailable."
    return "Selected products prefer known point coverage and diverse incidence/sun geometry."


def _preferred_product_set(product_ids: list[str] | None) -> set[str]:
    return {item for item in (_normalize_catalog_product_id(value) for value in (product_ids or [])) if item}


def _prioritize_preferred_products(df: Any, preferred_product_ids: list[str] | None, *, max_products: int) -> Any:
    """Move footprint-confirmed official catalog image IDs to the front.

    This does not invent coverage. Products are only boosted if the search
    already returned them as footprint hits for the deterministic tile.
    """
    if df is None or df.empty or "product_id" not in df or not preferred_product_ids:
        return df
    preferred = [_normalize_catalog_product_id(value) for value in preferred_product_ids]
    preferred = [value for value in preferred if value]
    if not preferred:
        return df
    ordered_indices: list[Any] = []
    used: set[Any] = set()
    product_series = df["product_id"].map(lambda value: _normalize_catalog_product_id(str(value)) or str(value))
    for product_id in preferred:
        matches = df.index[product_series == product_id].tolist()
        for index in matches:
            if index not in used:
                ordered_indices.append(index)
                used.add(index)
                break
    for index in df.index:
        if index not in used:
            ordered_indices.append(index)
            used.add(index)
    return df.loc[ordered_indices].head(max_products).reset_index(drop=True)


def _normalize_catalog_product_id(value: str) -> str | None:
    text = str(value).strip().upper()
    if not text:
        return None
    match = re.search(r"M\d+[LR](?:[EC])?", text)
    if not match:
        return None
    raw = match.group(0)
    if raw.endswith(("LE", "RE")):
        return raw
    if raw.endswith(("LC", "RC")):
        return raw[:-1] + "E"
    return raw + "E"


def _mapped_cache_dir(cache_root: Path, pixel_resolution: float) -> Path:
    """Shared full-strip cache keyed by map resolution.

    This prevents different nearby coordinates from rerunning the expensive
    full-strip ISIS mapping for the same NAC product during overnight runs.
    """
    safe_resolution = f"{float(pixel_resolution):.6f}".replace(".", "p")
    return cache_root / f"{safe_resolution}mpp"


def _write_or_validate_coordinate_lock(
    out_dir: Path,
    *,
    lat: float,
    lon: float,
    tile: LocationSearchTile,
    force: bool,
) -> None:
    """Prevent stale folders from silently switching target coordinates."""
    lock_path = out_dir / "coordinate_lock.json"
    pixel = pixel_in_tile(lat, lon, tile_size_m=tile.tile_size_m, meters_per_pixel=tile.ml_tile.meters_per_pixel)
    payload = {
        "lat": float(lat),
        "lon_original": float(lon),
        "lon_360": _normalize_lon_360(lon),
        "tile_id": tile.tile_id,
        "tile_i": int(tile.tile_x),
        "tile_j": int(tile.tile_y),
        "pixel_x": int(pixel["pixel_x"]),
        "pixel_y": int(pixel["pixel_y"]),
        "tile_size_m": float(tile.tile_size_m),
        "meters_per_pixel": float(tile.ml_tile.meters_per_pixel),
    }
    if lock_path.exists() and not force:
        existing = _read_json_if_exists(lock_path)
        mismatches = []
        for key in ("tile_id", "tile_i", "tile_j", "pixel_x", "pixel_y"):
            if existing.get(key) != payload.get(key):
                mismatches.append(f"{key}: existing={existing.get(key)} requested={payload.get(key)}")
        if _coordinate_delta_m(existing.get("lat"), existing.get("lon_original"), lat, lon) > 1.0:
            mismatches.append("target lat/lon differs by more than 1 meter")
        if mismatches:
            raise ValueError(
                f"Coordinate lock mismatch in {out_dir}. Refusing to reuse this folder: "
                + "; ".join(mismatches)
            )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _validate_crop_coordinate_lock(
    sidecar_path: Path,
    *,
    lat: float,
    lon: float,
    tile_size_m: float,
    meters_per_pixel: float,
) -> dict[str, Any]:
    """Verify a saved crop was made for the requested deterministic coordinate."""
    sidecar = _read_json_if_exists(sidecar_path)
    if not sidecar:
        return {"passed": False, "reason": f"missing crop sidecar {sidecar_path}"}
    expected = pixel_in_tile(lat, lon, tile_size_m=tile_size_m, meters_per_pixel=meters_per_pixel)
    target = sidecar.get("target", {})
    checks = {
        "tile_id": (sidecar.get("tile_id"), expected["tile_id"]),
        "tile_i": (sidecar.get("tile_i"), expected["tile_i"]),
        "tile_j": (sidecar.get("tile_j"), expected["tile_j"]),
        "pixel_x": (target.get("pixel_x"), expected["pixel_x"]),
        "pixel_y": (target.get("pixel_y"), expected["pixel_y"]),
    }
    mismatches = [
        f"{key}: actual={actual} expected={expected_value}"
        for key, (actual, expected_value) in checks.items()
        if actual != expected_value
    ]
    if _coordinate_delta_m(target.get("lat"), target.get("lon_original"), lat, lon) > 1.0:
        mismatches.append("sidecar target lat/lon differs by more than 1 meter")
    if mismatches:
        return {"passed": False, "reason": "; ".join(mismatches)}
    return {"passed": True, "reason": ""}


def _coordinate_delta_m(lat_a: Any, lon_a: Any, lat_b: Any, lon_b: Any) -> float:
    try:
        a_lat = math.radians(float(lat_a))
        b_lat = math.radians(float(lat_b))
        d_lat = b_lat - a_lat
        d_lon = math.radians(float(lon_b) - float(lon_a))
    except (TypeError, ValueError):
        return float("inf")
    mean_lat = (a_lat + b_lat) / 2.0
    return 1737400.0 * math.sqrt(d_lat * d_lat + (math.cos(mean_lat) * d_lon) ** 2)


def _process_and_crop_product(
    product_id: str,
    *,
    lat: float,
    lon: float,
    tile_size_km: float,
    pixel_resolution: float,
    img_url: str | None,
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
    crop_tif = images_dir / f"{product_id}.tif"
    crop_preview = images_dir / f"{product_id}.png"
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
                if not mapped_tif.exists():
                    completed = _run_product_map_cam2map(
                        product_id,
                        pixel_resolution=pixel_resolution,
                        img_url=img_url,
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
                _normalize_crop_output_names(crop_tif)
            else:
                return ProcessedNacProduct(product_id=product_id, status="failed", error=f"Unknown tile method: {tile_method}")

        if not crop_tif.exists():
            return ProcessedNacProduct(product_id=product_id, status="missing_output", error=f"Expected output missing: {crop_tif}")

        quality = _crop_quality_metrics(crop_tif)
        valid_fraction = float(quality.get("valid_pixel_fraction", 0.0))
        if valid_fraction < 0.80:
            _delete_crop_outputs(crop_tif)
            return ProcessedNacProduct(
                product_id=product_id,
                status="insufficient_valid_pixels",
                error=f"Only {valid_fraction:.3f} of the deterministic tile has valid pixels.",
            )
        if not bool(quality.get("passes_artifact_prefilter", False)):
            _delete_crop_outputs(crop_tif)
            reason = str(quality.get("artifact_rejection_reason", "failed artifact prefilter"))
            return ProcessedNacProduct(product_id=product_id, status="artifact_prefilter_rejected", error=reason)
        coordinate_check = _validate_crop_coordinate_lock(
            crop_tif.with_suffix(".json"),
            lat=lat,
            lon=lon,
            tile_size_m=tile_size_m,
            meters_per_pixel=pixel_resolution,
        )
        if not coordinate_check["passed"]:
            _delete_crop_outputs(crop_tif)
            return ProcessedNacProduct(
                product_id=product_id,
                status="coordinate_lock_failed",
                error=str(coordinate_check["reason"]),
            )

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


def _crop_valid_pixel_fraction(tif_path: Path) -> float:
    return float(_crop_quality_metrics(tif_path).get("valid_pixel_fraction", 0.0))


def _crop_quality_metrics(tif_path: Path) -> dict[str, Any]:
    try:
        import numpy as np
        import rasterio

        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype("float64", copy=False)
            valid = np.isfinite(arr)
            if src.nodata is not None:
                valid &= arr != src.nodata
            valid_fraction = float(valid.mean()) if valid.size else 0.0
            if not valid.any():
                return {
                    "valid_pixel_fraction": valid_fraction,
                    "passes_artifact_prefilter": False,
                    "artifact_rejection_reason": "no valid finite pixels",
                }
            values = arr[valid]
            p2, p98 = np.percentile(values, [2.0, 98.0])
            contrast = float(p98 - p2)
            pixel_std = float(np.std(values))
            # Conservative: reject only tiles with so little calibrated signal
            # that fabric/stripe detector noise can dominate the visual texture.
            min_contrast = 0.003
            min_std = 0.00075
            if contrast < min_contrast or pixel_std < min_std:
                return {
                    "valid_pixel_fraction": valid_fraction,
                    "pixel_std": pixel_std,
                    "contrast_p98_p2": contrast,
                    "passes_artifact_prefilter": False,
                    "artifact_rejection_reason": f"low calibrated contrast/std ({contrast:.6g}, {pixel_std:.6g})",
                }
            return {
                "valid_pixel_fraction": valid_fraction,
                "pixel_std": pixel_std,
                "contrast_p98_p2": contrast,
                "passes_artifact_prefilter": True,
                "artifact_rejection_reason": "",
            }
    except Exception:
        return {
            "valid_pixel_fraction": 0.0,
            "passes_artifact_prefilter": False,
            "artifact_rejection_reason": "could not read tile quality metrics",
        }


def _effective_product_pixel_resolution(row: Any, *, default_pixel_resolution: float, tile_size_m: float) -> float:
    """Return a near-native map/crop resolution that fits the deterministic tile.

    The production tile grid is physical-space deterministic, so the tile bounds
    do not depend on pixels. For visual quality, avoid forcing every NAC to
    0.5 m/px; use the product's native ODE/LROC map resolution when available,
    adjusted only enough that tile_size_m / meters_per_pixel is an integer.
    """
    native = None
    for column in ("resolution_m_per_pixel", "resolution", "RESOLUTION"):
        try:
            value = row.get(column)
        except AttributeError:
            value = None
        if value is None:
            continue
        try:
            native = float(value)
            if native > 0:
                break
        except (TypeError, ValueError):
            native = None
    if native is None or native <= 0:
        native = float(default_pixel_resolution)
    pixels = max(1, int(round(float(tile_size_m) / native)))
    return float(tile_size_m) / float(pixels)


def _delete_crop_outputs(crop_tif: Path) -> None:
    for path in (
        crop_tif,
        crop_tif.with_suffix(".tif.msk"),
        crop_tif.with_suffix(".png"),
        crop_tif.with_suffix(".json"),
        crop_tif.with_name(f"{crop_tif.stem}_preview.png"),
    ):
        path.unlink(missing_ok=True)


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
    img_url: str | None,
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
    if img_url:
        cmd.extend(["--img-url", img_url])
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
        "fixed",
    ]
    return subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)


def _normalize_crop_output_names(crop_tif: Path) -> None:
    """Keep location NAC image names compact and consistent."""
    generated_preview = crop_tif.with_name(f"{crop_tif.stem}_preview.png")
    compact_preview = crop_tif.with_suffix(".png")
    if generated_preview.exists() and generated_preview != compact_preview:
        generated_preview.replace(compact_preview)


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
    parser.add_argument("--pixel-resolution", type=float, default=1.0)
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
    print(f"location tile: {out_dir / 'tile.json'}")
    print(f"location audit: {out_dir / 'audit.json'}")
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
