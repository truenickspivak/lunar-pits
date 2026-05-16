"""LROC NAC EDR footprint matching and planned crop outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

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
RESOLUTION_SIMILAR_RELATIVE_TOLERANCE = 0.50
RESOLUTION_SIMILAR_ABSOLUTE_TOLERANCE_M = 0.50
MIN_USABLE_INCIDENCE_DEG = 5.0
MAX_USABLE_INCIDENCE_DEG = 90.0
MAX_USABLE_EMISSION_DEG = 45.0
MIN_USABLE_PHASE_DEG = 10.0
MAX_USABLE_PHASE_DEG = 120.0


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
        - work["_resolution_rank"] * 100.0
        - work["resolution_m_per_pixel"].fillna(99.0) * 10.0
    )
    ranked = work.sort_values(
        ["_coverage_tier", "_resolution_rank", "resolution_m_per_pixel", "product_id"],
        ascending=[True, True, True, True],
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
                    incidence_gap * 10.0,
                    float(row.get("tile_coverage_fraction") or 0.0),
                    -resolution_delta,
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
        products[products["full_tile_candidate"] & ~products["usable_observation_geometry"]].copy(),
        products[(~products["full_tile_candidate"]) & (products["near_full_tile_candidate"]) & products["usable_observation_geometry"]].copy(),
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
