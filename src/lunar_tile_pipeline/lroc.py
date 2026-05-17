"""LROC NAC EDR footprint matching and planned crop outputs."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely import wkt

from lunar_tile_pipeline.footprints import load_footprints, tile_center_point_for_footprints, tile_polygon_for_footprints
from lunar_tile_pipeline.tiling import LunarTile


PRODUCT_ID_COLUMNS = ("product_id", "PRODUCT_ID", "ProductId", "prod_id", "PROD_ID")
START_TIME_COLUMNS = ("start_time", "START_TIME", "observation_time")
INCIDENCE_COLUMNS = ("incidence_angle", "INC_ANGLE", "INCIDENCE")
EMISSION_COLUMNS = ("emission_angle", "EMSSN_ANG", "EMISSION")
PHASE_COLUMNS = ("phase_angle", "PHASE_ANGL", "PHASE")
RESOLUTION_COLUMNS = ("resolution_m_per_pixel", "RESOLUTION", "resolution")
FULL_TILE_COVERAGE_FRACTION = 0.95
NEAR_FULL_TILE_COVERAGE_FRACTION = 0.75
RESOLUTION_SIMILAR_RELATIVE_TOLERANCE = 0.35
RESOLUTION_SIMILAR_ABSOLUTE_TOLERANCE_M = 0.20
MIN_USABLE_INCIDENCE_DEG = 5.0
MAX_USABLE_INCIDENCE_DEG = 90.0
MAX_USABLE_EMISSION_DEG = 45.0
MAX_ALIGNED_EMISSION_DEG = 15.0
MIN_USABLE_PHASE_DEG = 10.0
MAX_USABLE_PHASE_DEG = 120.0
ODE_REST_URL = "https://oderest.rsl.wustl.edu/live2"


def parse_lroc_product_id(product_id: str) -> dict[str, str | None]:
    normalized = str(product_id).strip().upper()
    match = re.fullmatch(r"M\d+([LR])([EC])", normalized)
    camera = None
    processing_level = "unknown"
    if match:
        camera = "NAC-L" if match.group(1) == "L" else "NAC-R"
        processing_level = "EDR" if match.group(2) == "E" else "CDR"
    return {"product_id": normalized, "camera": camera, "processing_level": processing_level}


def _first_value(row: pd.Series, candidates: tuple[str, ...]) -> Any:
    lookup = {str(col).lower(): col for col in row.index}
    for candidate in candidates:
        col = lookup.get(candidate.lower())
        if col is not None and pd.notna(row[col]):
            return row[col]
    return None


def _edr_id_from_row(row: pd.Series) -> str | None:
    raw = _first_value(row, PRODUCT_ID_COLUMNS)
    if raw is None:
        return None
    parsed = parse_lroc_product_id(str(raw))
    product_id = parsed["product_id"]
    if parsed["processing_level"] == "EDR":
        return product_id
    if parsed["processing_level"] == "CDR":
        return f"{product_id[:-1]}E"
    return None


def _row_metadata(
    row: pd.Series,
    source_path: str,
    contains_center: bool,
    intersects_tile: bool,
    area: float,
    tile_area: float,
) -> dict[str, Any]:
    product_id = _edr_id_from_row(row)
    if product_id is None:
        raise ValueError("Row is not an LROC NAC EDR-compatible product")
    parsed = parse_lroc_product_id(product_id)
    coverage_fraction = min(float(area) / tile_area, 1.0) if tile_area > 0 else 0.0
    return {
        "product_id": product_id,
        "camera": parsed["camera"],
        "processing_level": parsed["processing_level"],
        "contains_center": bool(contains_center),
        "intersects_tile": bool(intersects_tile),
        "intersection_area_approx": float(area),
        "tile_area_approx": float(tile_area),
        "tile_coverage_fraction": float(coverage_fraction),
        "full_tile_candidate": bool(coverage_fraction >= FULL_TILE_COVERAGE_FRACTION),
        "start_time": _first_value(row, START_TIME_COLUMNS),
        "incidence_angle": _first_value(row, INCIDENCE_COLUMNS),
        "emission_angle": _first_value(row, EMISSION_COLUMNS),
        "phase_angle": _first_value(row, PHASE_COLUMNS),
        "resolution_m_per_pixel": _first_value(row, RESOLUTION_COLUMNS),
        "source_footprint_file": source_path,
    }


def is_lroc_nac_edr_row(row: pd.Series) -> bool:
    product_id = _edr_id_from_row(row)
    if product_id is None:
        return False
    parsed = parse_lroc_product_id(product_id)
    return parsed["camera"] in {"NAC-L", "NAC-R"} and parsed["processing_level"] == "EDR"


def find_lroc_nac_edr_for_tile(tile: LunarTile, footprints_gdf: gpd.GeoDataFrame, source_footprint_file: str = "") -> pd.DataFrame:
    if footprints_gdf.empty:
        return pd.DataFrame()
    tile_polygon = tile_polygon_for_footprints(tile, footprints_gdf)
    tile_center = tile_center_point_for_footprints(tile, footprints_gdf)
    tile_area = float(tile_polygon.area)
    spatial_index = footprints_gdf.sindex
    candidate_indices = list(spatial_index.query(tile_polygon, predicate="intersects"))
    candidates = footprints_gdf.iloc[candidate_indices] if candidate_indices else footprints_gdf.iloc[[]]
    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        if not is_lroc_nac_edr_row(row):
            continue
        geom = row.geometry
        intersects_tile = bool(geom.intersects(tile_polygon))
        if not intersects_tile:
            continue
        contains_center = bool(geom.contains(tile_center) or geom.touches(tile_center))
        area = geom.intersection(tile_polygon).area
        rows.append(_row_metadata(row, source_footprint_file, contains_center, intersects_tile, area, tile_area))
    return pd.DataFrame(rows).drop_duplicates(subset=["product_id"]) if rows else pd.DataFrame()


def find_lroc_nac_edr_for_tile_from_file(tile: LunarTile, footprints_path: str | Path) -> pd.DataFrame:
    gdf = load_footprints(footprints_path)
    return find_lroc_nac_edr_for_tile(tile, gdf, source_footprint_file=str(footprints_path))


def find_lroc_nac_edr_for_tile_from_ode(
    tile: LunarTile,
    *,
    limit: int = 80,
    timeout: int = 90,
    exact_coverage: bool = False,
) -> pd.DataFrame:
    """Find LROC NAC EDR products intersecting a tile using ODE REST.

    The old LROC ``NAC_EQ_SCIENCE_MISSION_360`` shapefile is useful, but it is
    not complete enough for the pit catalog or whole-Moon scans. ODE's product
    endpoint with ``pt=EDRNAC4`` is the broader footprint-aware source: it
    returns PDS4 LROC NAC EDR products whose footprints intersect the requested
    lat/lon box, plus image URL, timing, angles, resolution, and footprint WKT.
    """
    bounds = tile.bounds_latlon
    min_lat = float(bounds["min_lat_approx"])
    max_lat = float(bounds["max_lat_approx"])
    min_lon = float(bounds["min_lon_360_approx"])
    max_lon = float(bounds["max_lon_360_approx"])
    if bool(bounds.get("crosses_lon_360_wrap")):
        parts = [(min_lon, 360.0), (0.0, max_lon)]
    else:
        parts = [(min_lon, max_lon)]

    rows: list[dict[str, Any]] = []
    for west, east in parts:
        rows.extend(_ode_products_for_bbox(min_lat, max_lat, west, east, limit=limit, timeout=timeout))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["product_id"])
    df = _add_ode_tile_coverage(df, tile) if exact_coverage else _add_ode_bbox_tile_coverage(df, tile)
    return rank_lroc_nac_edr_for_tile(df)


def _ode_products_for_bbox(min_lat: float, max_lat: float, west_lon: float, east_lon: float, *, limit: int, timeout: int) -> list[dict[str, Any]]:
    params = {
        "query": "product",
        "target": "moon",
        "results": "fopm",
        "output": "JSON",
        "ihid": "LRO",
        "iid": "LROC",
        "pt": "EDRNAC4",
        "minlat": f"{min_lat:.8f}",
        "maxlat": f"{max_lat:.8f}",
        "westernlon": f"{west_lon:.8f}",
        "easternlon": f"{east_lon:.8f}",
        "loc": "i",
        "limit": str(int(limit)),
    }
    url = f"{ODE_REST_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    results = payload.get("ODEResults", {})
    if results.get("Status") == "ERROR":
        raise RuntimeError(str(results.get("Error", "ODE REST query failed")))
    products = results.get("Products", {}).get("Product", [])
    if isinstance(products, dict):
        products = [products]
    records: list[dict[str, Any]] = []
    for product in products:
        record = _ode_product_to_record(product, url)
        if record is not None:
            records.append(record)
    return records


def _ode_product_to_record(product: dict[str, Any], source_url: str) -> dict[str, Any] | None:
    product_id = _ode_product_id(product)
    if product_id is None:
        return None
    parsed = parse_lroc_product_id(product_id)
    if parsed["camera"] not in {"NAC-L", "NAC-R"} or parsed["processing_level"] != "EDR":
        return None
    footprint_wkt = str(product.get("Footprint_geometry") or product.get("Footprint_GL_geometry") or "").strip()
    geometry = None
    if footprint_wkt:
        try:
            geometry = wkt.loads(footprint_wkt)
        except Exception:
            geometry = None
    return {
        "product_id": product_id,
        "camera": parsed["camera"],
        "processing_level": parsed["processing_level"],
        "contains_center": False,
        "intersects_tile": True,
        "intersection_area_approx": None,
        "tile_area_approx": None,
        "tile_coverage_fraction": 1.0,
        "full_tile_candidate": True,
        "start_time": product.get("UTC_start_time") or product.get("Observation_time"),
        "incidence_angle": _maybe_float(product.get("Incidence_angle")),
        "emission_angle": _maybe_float(product.get("Emission_angle")),
        "phase_angle": _maybe_float(product.get("Phase_angle")),
        "resolution_m_per_pixel": _maybe_float(product.get("Map_resolution")),
        "source_footprint_file": source_url,
        "footprint_source": "ODE REST EDRNAC4 footprint intersection",
        "coverage_method": "ode_footprint_intersects_tile",
        "img_url": _ode_file_url(product, ".IMG"),
        "label_url": product.get("LabelURL") or _ode_file_url(product, ".XML"),
        "browse_url": _ode_browse_url(product),
        "volume": product.get("PDSVolume_Id"),
        "relative_path": product.get("RelativePathtoVol"),
        "minimum_latitude": _maybe_float(product.get("Minimum_latitude")),
        "maximum_latitude": _maybe_float(product.get("Maximum_latitude")),
        "westernmost_longitude": _maybe_float(product.get("Westernmost_longitude")),
        "easternmost_longitude": _maybe_float(product.get("Easternmost_longitude")),
        "footprints_cross_meridian": product.get("Footprints_cross_meridian"),
        "footprint_wkt": footprint_wkt,
        "footprint_area_approx": float(geometry.area) if geometry is not None else None,
        "ode_id": product.get("ode_id"),
    }


def _ode_product_id(product: dict[str, Any]) -> str | None:
    for key in ("pdsid", "LabelFileName", "Product_name"):
        value = product.get(key)
        if not value:
            continue
        text = str(value).strip().upper()
        if "." in text:
            text = text.split(".")[-1]
        if text.endswith(".IMG") or text.endswith(".XML"):
            text = text.rsplit(".", 1)[0]
        match = re.search(r"M\d+[LR][EC]", text)
        if match:
            parsed = parse_lroc_product_id(match.group(0))
            if parsed["processing_level"] == "EDR":
                return parsed["product_id"]
            if parsed["processing_level"] == "CDR":
                return f"{parsed['product_id'][:-1]}E"
    return None


def _ode_files(product: dict[str, Any]) -> list[dict[str, Any]]:
    files = product.get("Product_files", {}).get("Product_file", [])
    if isinstance(files, dict):
        return [files]
    return files if isinstance(files, list) else []


def _ode_file_url(product: dict[str, Any], suffix: str) -> str | None:
    suffix = suffix.upper()
    for item in _ode_files(product):
        name = str(item.get("FileName", "")).upper()
        if name.endswith(suffix) and str(item.get("Type", "")).lower() == "product":
            return item.get("URL")
    return None


def _ode_browse_url(product: dict[str, Any]) -> str | None:
    for item in _ode_files(product):
        if str(item.get("Type", "")).lower() == "browse":
            return item.get("URL")
    return None


def _add_ode_bbox_tile_coverage(products: pd.DataFrame, tile: LunarTile) -> pd.DataFrame:
    """Cheap tile-coverage estimate from ODE's product lat/lon bounds.

    ODE already does the authoritative footprint intersection query. Parsing
    and intersecting every returned WKT polygon is useful for diagnostics, but
    it is avoidable overhead in the normal gatherer path. The final raster tile
    QA is still the hard gate for black/partial products.
    """
    if products.empty:
        return products
    out = products.copy()
    bounds = tile.bounds_latlon
    tile_min_lat = float(bounds["min_lat_approx"])
    tile_max_lat = float(bounds["max_lat_approx"])
    tile_min_lon = float(bounds["min_lon_360_approx"])
    tile_max_lon = float(bounds["max_lon_360_approx"])
    tile_crosses_wrap = bool(bounds.get("crosses_lon_360_wrap"))
    tile_lat_span = max(tile_max_lat - tile_min_lat, 1e-12)
    tile_lon_span = _lon_interval_span(tile_min_lon, tile_max_lon, tile_crosses_wrap)
    tile_area = tile_lat_span * max(tile_lon_span, 1e-12)

    fractions: list[float] = []
    for _, row in out.iterrows():
        p_min_lat = _maybe_float(row.get("minimum_latitude"))
        p_max_lat = _maybe_float(row.get("maximum_latitude"))
        p_west = _maybe_float(row.get("westernmost_longitude"))
        p_east = _maybe_float(row.get("easternmost_longitude"))
        crosses = _truthy(row.get("footprints_cross_meridian"))
        if None in (p_min_lat, p_max_lat, p_west, p_east):
            fractions.append(1.0)
            continue
        lat_overlap = max(0.0, min(tile_max_lat, p_max_lat) - max(tile_min_lat, p_min_lat))
        lon_overlap = _lon_interval_overlap(tile_min_lon, tile_max_lon, tile_crosses_wrap, p_west, p_east, crosses)
        fraction = max(0.0, min((lat_overlap * lon_overlap) / tile_area, 1.0))
        fractions.append(float(fraction))

    out["intersection_area_approx"] = [float(value * tile_area) for value in fractions]
    out["tile_area_approx"] = float(tile_area)
    out["tile_coverage_fraction"] = fractions
    out["full_tile_candidate"] = out["tile_coverage_fraction"] >= FULL_TILE_COVERAGE_FRACTION
    out["coverage_estimate_method"] = "ode_product_bbox_fast"
    return out


def _add_ode_tile_coverage(products: pd.DataFrame, tile: LunarTile) -> pd.DataFrame:
    """Compute true tile overlap for ODE WKT footprints instead of assuming 1.0."""
    if products.empty or "footprint_wkt" not in products.columns:
        return products
    out = products.copy()
    tile_polygon = _tile_polygon_360(tile)
    tile_area = float(tile_polygon.area)
    fractions: list[float] = []
    areas: list[float] = []
    contains_center: list[bool] = []
    center = tile_center_point_for_footprints(tile, _dummy_360_gdf())
    for value in out["footprint_wkt"].fillna(""):
        try:
            geom = wkt.loads(str(value))
            area = float(geom.intersection(tile_polygon).area)
            fraction = min(area / tile_area, 1.0) if tile_area > 0 else 0.0
            contains = bool(geom.contains(center) or geom.touches(center))
        except Exception:
            area = 0.0
            fraction = 0.0
            contains = False
        areas.append(area)
        fractions.append(fraction)
        contains_center.append(contains)
    out["intersection_area_approx"] = areas
    out["tile_area_approx"] = tile_area
    out["tile_coverage_fraction"] = fractions
    out["full_tile_candidate"] = out["tile_coverage_fraction"] >= FULL_TILE_COVERAGE_FRACTION
    out["contains_center"] = contains_center
    out["coverage_estimate_method"] = "ode_wkt_exact"
    return out


def _lon_interval_span(west: float, east: float, crosses: bool) -> float:
    if crosses or east < west:
        return (360.0 - west) + east
    return east - west


def _lon_intervals(west: float, east: float, crosses: bool) -> list[tuple[float, float]]:
    west = west % 360.0
    east = east % 360.0
    if crosses or east < west:
        return [(west, 360.0), (0.0, east)]
    return [(west, east)]


def _lon_interval_overlap(a_west: float, a_east: float, a_crosses: bool, b_west: float, b_east: float, b_crosses: bool) -> float:
    overlap = 0.0
    for aw, ae in _lon_intervals(a_west, a_east, a_crosses):
        for bw, be in _lon_intervals(b_west, b_east, b_crosses):
            overlap += max(0.0, min(ae, be) - max(aw, bw))
    return overlap


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _tile_polygon_360(tile: LunarTile):
    from shapely.geometry import Polygon

    coords = []
    for item in tile.bounds_latlon["corner_latlon"]:
        coords.append((float(item["lon_360"]), float(item["lat"])))
    return Polygon(coords)


def _dummy_360_gdf() -> gpd.GeoDataFrame:
    from shapely.geometry import Point

    return gpd.GeoDataFrame(geometry=[Point(0, 0), Point(360, 0)])


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def enrich_lroc_metadata(products: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for optional ODE/PDS metadata enrichment."""
    if products.empty:
        return products
    enriched = products.copy()
    enriched["enrichment_status"] = "not_implemented"
    enriched["enrichment_note"] = "Optional ODE/PDS metadata enrichment is planned but not implemented in this first version."
    return enriched


def rank_lroc_nac_edr_for_tile(products: pd.DataFrame) -> pd.DataFrame:
    """Rank tile footprint hits for fixed-tile processing.

    The priority order is deliberately conservative:
    full or nearly full deterministic-tile coverage first, best/similar resolution
    second, then incidence diversity within that high-quality candidate set.
    """
    if products.empty:
        return products.copy()

    work = products.drop_duplicates(subset=["product_id"]).copy()
    for col in ("tile_coverage_fraction", "resolution_m_per_pixel", "incidence_angle", "emission_angle", "phase_angle"):
        if col not in work.columns:
            work[col] = pd.NA
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["tile_coverage_fraction"] = work["tile_coverage_fraction"].fillna(0.0).clip(lower=0.0, upper=1.0)
    work["full_tile_candidate"] = work["tile_coverage_fraction"] >= FULL_TILE_COVERAGE_FRACTION
    work["near_full_tile_candidate"] = work["tile_coverage_fraction"] >= NEAR_FULL_TILE_COVERAGE_FRACTION
    work["usable_observation_geometry"] = work.apply(_has_usable_observation_geometry, axis=1)
    work["aligned_observation_geometry"] = work.apply(_has_aligned_observation_geometry, axis=1)

    def coverage_tier(value: float) -> int:
        if value >= FULL_TILE_COVERAGE_FRACTION:
            return 0
        if value >= NEAR_FULL_TILE_COVERAGE_FRACTION:
            return 1
        return 2

    work["_coverage_tier"] = work["tile_coverage_fraction"].map(coverage_tier)
    finite_resolution = work["resolution_m_per_pixel"].dropna()
    best_resolution = float(finite_resolution.min()) if not finite_resolution.empty else None
    if best_resolution is None:
        work["resolution_similarity_group"] = "unknown"
        work["_resolution_rank"] = 0
    else:
        tolerance = max(
            best_resolution * RESOLUTION_SIMILAR_RELATIVE_TOLERANCE,
            RESOLUTION_SIMILAR_ABSOLUTE_TOLERANCE_M,
        )
        work["resolution_similarity_group"] = work["resolution_m_per_pixel"].map(
            lambda value: "similar_best" if pd.notna(value) and float(value) <= best_resolution + tolerance else "coarser"
        )
        work["_resolution_rank"] = work["resolution_similarity_group"].map({"similar_best": 0, "coarser": 1}).fillna(2)

    work["tile_rank_score"] = (
        work["tile_coverage_fraction"] * 1000.0
        + work["usable_observation_geometry"].astype(float) * 250.0
        + work["aligned_observation_geometry"].astype(float) * 25.0
        - work["_resolution_rank"] * 100.0
        - work["resolution_m_per_pixel"].fillna(99.0) * 10.0
    )
    ranked = work.sort_values(
        ["_coverage_tier", "usable_observation_geometry", "_resolution_rank", "resolution_m_per_pixel", "product_id"],
        ascending=[True, False, True, True, True],
        na_position="last",
    ).copy()
    ranked["tile_processing_rank"] = range(1, len(ranked) + 1)
    return ranked.drop(columns=[c for c in ranked.columns if c.startswith("_")], errors="ignore")


def select_top_lroc_nac_for_tile(products: pd.DataFrame, max_products: int = 3) -> pd.DataFrame:
    """Select top tile products while preserving incidence diversity."""
    ranked = rank_lroc_nac_edr_for_tile(products)
    if ranked.empty or max_products <= 0:
        return ranked.head(0)

    candidate_source = ranked.copy()

    selected_indices: list[int] = []

    for coverage_pool in _selection_pools(candidate_source):
        if len(selected_indices) >= max_products:
            break
        coverage_pool = coverage_pool.drop(index=selected_indices, errors="ignore")
        if coverage_pool.empty:
            continue

        if coverage_pool["full_tile_candidate"].any() or coverage_pool["near_full_tile_candidate"].any():
            anchor = coverage_pool.sort_values(
                ["resolution_m_per_pixel", "tile_coverage_fraction", "product_id"],
                ascending=[True, False, True],
                na_position="last",
            ).iloc[0]
        else:
            anchor = coverage_pool.sort_values(
                ["tile_coverage_fraction", "resolution_m_per_pixel", "product_id"],
                ascending=[False, True, True],
                na_position="last",
            ).iloc[0]

        anchor_resolution = anchor.get("resolution_m_per_pixel")
        if pd.notna(anchor_resolution):
            tolerance = max(
                float(anchor_resolution) * RESOLUTION_SIMILAR_RELATIVE_TOLERANCE,
                RESOLUTION_SIMILAR_ABSOLUTE_TOLERANCE_M,
            )
            similar_resolution = coverage_pool[
                coverage_pool["resolution_m_per_pixel"].isna()
                | ((coverage_pool["resolution_m_per_pixel"] - float(anchor_resolution)).abs() <= tolerance)
            ].copy()
            slots_left = max_products - len(selected_indices)
            if len(similar_resolution) >= slots_left:
                coverage_pool = similar_resolution

        ordered = coverage_pool.sort_values(
            ["tile_coverage_fraction", "resolution_m_per_pixel", "product_id"],
            ascending=[False, True, True],
            na_position="last",
        )

        while len(selected_indices) < max_products and len(selected_indices) < len(candidate_source):
            remaining = ordered.drop(index=selected_indices, errors="ignore")
            if remaining.empty:
                break
            if not selected_indices:
                selected_indices.append(remaining.index[0])
                continue
            selected_incidence = candidate_source.loc[selected_indices, "incidence_angle"]

            def choice_score(row: pd.Series) -> tuple[float, float, float, str]:
                if pd.isna(row.get("incidence_angle")) or selected_incidence.dropna().empty:
                    incidence_gap = 0.0
                else:
                    incidence_gap = min(abs(float(row["incidence_angle"]) - float(value)) for value in selected_incidence.dropna())
                resolution_delta = 0.0
                if pd.notna(row.get("resolution_m_per_pixel")) and pd.notna(anchor_resolution):
                    resolution_delta = abs(float(row["resolution_m_per_pixel"]) - float(anchor_resolution))
                return (
                    float(row.get("tile_coverage_fraction") or 0.0),
                    -resolution_delta,
                    incidence_gap,
                    str(row["product_id"]),
                )

            best = max(remaining.iterrows(), key=lambda item: choice_score(item[1]))
            selected_indices.append(best[0])

    selected = candidate_source.loc[selected_indices].copy()
    selected["selected_for_processing_rank"] = range(1, len(selected) + 1)
    return selected


def _selection_pools(products: pd.DataFrame) -> list[pd.DataFrame]:
    pools = [
        products[products["full_tile_candidate"] & products["usable_observation_geometry"]].copy(),
        products[products["full_tile_candidate"] & products["aligned_observation_geometry"]].copy(),
        products[products["full_tile_candidate"] & ~products["usable_observation_geometry"]].copy(),
        products[(~products["full_tile_candidate"]) & (products["near_full_tile_candidate"]) & products["usable_observation_geometry"]].copy(),
        products[(~products["full_tile_candidate"]) & (products["near_full_tile_candidate"]) & products["aligned_observation_geometry"]].copy(),
        products[(~products["full_tile_candidate"]) & (products["near_full_tile_candidate"]) & ~products["usable_observation_geometry"]].copy(),
        products[(~products["full_tile_candidate"]) & (~products["near_full_tile_candidate"]) & products["usable_observation_geometry"]].copy(),
        products[(~products["full_tile_candidate"]) & (~products["near_full_tile_candidate"]) & ~products["usable_observation_geometry"]].copy(),
    ]
    return [pool for pool in pools if not pool.empty]


def _has_usable_observation_geometry(row: pd.Series) -> bool:
    incidence = row.get("incidence_angle")
    if pd.notna(incidence) and not (MIN_USABLE_INCIDENCE_DEG <= float(incidence) <= MAX_USABLE_INCIDENCE_DEG):
        return False
    emission = row.get("emission_angle")
    if pd.notna(emission) and float(emission) > MAX_USABLE_EMISSION_DEG:
        return False
    phase = row.get("phase_angle")
    if pd.notna(phase) and not (MIN_USABLE_PHASE_DEG <= float(phase) <= MAX_USABLE_PHASE_DEG):
        return False
    return True


def _has_aligned_observation_geometry(row: pd.Series) -> bool:
    """Return true for products likely to keep steep pit features close to ground coordinates."""
    emission = row.get("emission_angle")
    return pd.isna(emission) or float(emission) <= MAX_ALIGNED_EMISSION_DEG


def crop_nac_to_tile(product_id: str, tile: LunarTile, output_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    planned_tif = out_dir / f"{product_id}_{tile.tile_id}.tif"
    metadata = {
        "product_id": product_id,
        "tile_id": tile.tile_id,
        "planned_output": str(planned_tif),
        "status": "planned_not_implemented",
        "todo": "Locate/download NAC EDR, project into the fixed tile frame, and save a tile-aligned raster.",
    }
    (out_dir / f"{product_id}_{tile.tile_id}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
