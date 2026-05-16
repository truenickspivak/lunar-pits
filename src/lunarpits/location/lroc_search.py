"""Search LROC NAC EDR footprints for location-context products."""

from __future__ import annotations

import csv
import io
import re
import zipfile
import urllib.request
from pathlib import Path
from typing import Iterable

from lunarpits.processing.identifiers import normalize_product_id


PDS_ROOT = "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-2-EDR-V1.0"
DEFAULT_CACHE_DIR = Path("data/cache/lroc_collections")
DEFAULT_FOOTPRINT_CACHE_DIR = Path("data/cache/lroc_footprints")
MOON_RADIUS_M = 1_737_400.0
NAC_EQ_SCIENCE_360_URL = (
    "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/"
    "EXTRAS/SHAPEFILE/NAC_EQ_SCIENCE_MISSION/NAC_EQ_SCIENCE_MISSION_360.ZIP"
)

PRODUCT_PATTERN = re.compile(r"\bM\d{6,}[LR]E\b", re.IGNORECASE)
IMG_PATH_PATTERN = re.compile(
    r"(?:DATA/)?(?P<relative>(?:MAP|COM|SCI)/[^,\"']*?/NAC/(?P<product>M\d{6,}[LR]E)\.IMG)",
    re.IGNORECASE,
)

ANGLE_COLUMNS = {
    "incidence_angle": ("incidence_angle", "incidence", "incidenceangle"),
    "emission_angle": ("emission_angle", "emission", "emissionangle"),
    "phase_angle": ("phase_angle", "phase", "phaseangle"),
    "sun_azimuth": ("sun_azimuth", "solar_azimuth", "sunazimuth", "sub_solar_azimuth"),
    "resolution": ("resolution", "map_resolution", "pixel_resolution", "sample_resolution"),
}

FOOTPRINT_COLUMNS = {
    "min_lat": ("min_lat", "minimum_latitude", "minimumlatitude", "south_latitude"),
    "max_lat": ("max_lat", "maximum_latitude", "maximumlatitude", "north_latitude"),
    "min_lon": ("min_lon", "minimum_longitude", "minimumlongitude", "west_longitude"),
    "max_lon": ("max_lon", "maximum_longitude", "maximumlongitude", "east_longitude"),
}

FOOTPRINT_OUTPUT_COLUMNS = [
    "product_id",
    "source_product_id",
    "camera",
    "img_url",
    "source_url",
    "volume",
    "relative_path",
    "observation_time",
    "spacecraft_clock_start_count",
    "incidence_angle",
    "emission_angle",
    "phase_angle",
    "sun_azimuth",
    "resolution",
    "center_lat",
    "center_lon",
    "contains_point",
    "coverage_method",
    "quality_tier",
    "quality_score",
    "footprint_source",
    "footprint_wkt",
    "search_note",
]


def volume_names(max_volume: int = 80) -> list[str]:
    names: list[str] = []
    for i in range(1, max_volume + 1):
        names.append(f"LROLRC_{i:04d}")
        names.extend(f"LROLRC_{i:04d}{suffix}" for suffix in ("A", "B", "C", "D"))
    return names


def collection_csv_url(volume: str) -> str:
    lower = volume.lower()
    return f"{PDS_ROOT}/{volume}/DATA/collection_lro-l-lroc-2-edr_{lower}_data.csv"


def normalize_longitude_360(lon: float) -> float:
    """Normalize longitude to the 0-360 east convention used by LROC 360-domain footprints."""
    normalized = lon % 360.0
    if normalized < 0:
        normalized += 360.0
    return normalized


def km_to_lat_degrees(radius_km: float) -> float:
    return (radius_km * 1000.0 / MOON_RADIUS_M) * (180.0 / 3.141592653589793)


def km_to_lon_degrees(radius_km: float, lat: float) -> float:
    import math

    cos_lat = max(abs(math.cos(math.radians(lat))), 1e-6)
    return km_to_lat_degrees(radius_km) / cos_lat


def footprint_cache_zip_path(cache_dir: Path = DEFAULT_FOOTPRINT_CACHE_DIR) -> Path:
    return cache_dir / "NAC_EQ_SCIENCE_MISSION_360.ZIP"


def footprint_extract_dir(cache_dir: Path = DEFAULT_FOOTPRINT_CACHE_DIR) -> Path:
    return cache_dir / "NAC_EQ_SCIENCE_MISSION_360"


def ensure_nac_equatorial_footprints(
    *,
    cache_dir: str | Path = DEFAULT_FOOTPRINT_CACHE_DIR,
    refresh_cache: bool = False,
) -> Path:
    """Download/extract the LROC NAC equatorial science footprint shapefile and return the .shp path."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    zip_path = footprint_cache_zip_path(cache_path)
    extract_dir = footprint_extract_dir(cache_path)

    if refresh_cache and zip_path.exists():
        zip_path.unlink()
    if not zip_path.exists():
        urllib.request.urlretrieve(NAC_EQ_SCIENCE_360_URL, zip_path)

    shp_files = list(extract_dir.rglob("*.shp")) if extract_dir.exists() else []
    if refresh_cache or not shp_files:
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        shp_files = list(extract_dir.rglob("*.shp"))

    if not shp_files:
        raise RuntimeError(f"No shapefile found after extracting {zip_path}")
    return shp_files[0]


def _empty_footprint_dataframe(warnings: list[str] | None = None):
    import pandas as pd

    df = pd.DataFrame(columns=FOOTPRINT_OUTPUT_COLUMNS + ["selected"])
    df.attrs["warnings"] = warnings or []
    return df


def _metadata_value(row: object, *names: str) -> object | None:
    for name in names:
        if name in row.index:
            value = row[name]
            if value is not None and str(value).strip() != "":
                return value
    return None


def _source_nac_product_to_edr_id(product_id: str) -> str | None:
    product_id = normalize_product_id(product_id)
    if re.fullmatch(r"M\d{6,}[LR]E", product_id):
        return product_id
    if re.fullmatch(r"M\d{6,}[LR]C", product_id):
        return f"{product_id[:-1]}E"
    return None


def _is_nac_footprint_product(product_id: str, row: object) -> bool:
    edr_id = _source_nac_product_to_edr_id(product_id)
    if edr_id is None:
        return False
    frame = str(_metadata_value(row, "NAC_FRM_ID", "camera") or "").upper()
    if frame and frame not in {"LEFT", "RIGHT", "L", "R", "NAC-L", "NAC-R"}:
        return False
    return True


def _camera_from_product(product_id: str, row: object) -> str | None:
    frame = str(_metadata_value(row, "NAC_FRM_ID") or "").upper()
    if "LEFT" in frame or product_id.endswith("LE"):
        return "NAC-L"
    if "RIGHT" in frame or product_id.endswith("RE"):
        return "NAC-R"
    return None


def _row_to_footprint_record(row: object, coverage_method: str, source: str) -> dict[str, object]:
    source_product_id = normalize_product_id(str(_metadata_value(row, "PRODUCT_ID", "product_id")))
    product_id = _source_nac_product_to_edr_id(source_product_id)
    if product_id is None:
        raise ValueError(f"Not a NAC EDR-compatible footprint product: {source_product_id}")
    url = _metadata_value(row, "URL", "img_url")
    relative_path = _metadata_value(row, "FILE_SPECI", "relative_path")
    incidence = _maybe_float(_metadata_value(row, "INC_ANGLE", "incidence_angle"))
    emission = _maybe_float(_metadata_value(row, "EMSSN_ANG", "emission_angle"))
    phase = _maybe_float(_metadata_value(row, "PHASE_ANGL", "phase_angle"))
    resolution = _maybe_float(_metadata_value(row, "RESOLUTION", "resolution"))
    quality_tier, quality_score = score_nac_metadata_quality(
        coverage_method=coverage_method,
        incidence_angle=incidence,
        emission_angle=emission,
        phase_angle=phase,
        resolution=resolution,
    )
    return {
        "product_id": product_id,
        "source_product_id": source_product_id,
        "camera": _camera_from_product(product_id, row),
        "img_url": None,
        "source_url": str(url) if url is not None else None,
        "volume": _metadata_value(row, "PDS_VOLUME", "volume"),
        "relative_path": str(relative_path).replace("\\", "/") if relative_path is not None else None,
        "observation_time": _metadata_value(row, "START_TIME", "observation_time"),
        "spacecraft_clock_start_count": _metadata_value(row, "SCLK_STCNT", "spacecraft_clock_start_count"),
        "incidence_angle": incidence,
        "emission_angle": emission,
        "phase_angle": phase,
        "sun_azimuth": _maybe_float(_metadata_value(row, "SUBSOLAZIM", "sun_azimuth")),
        "resolution": resolution,
        "center_lat": _maybe_float(_metadata_value(row, "CENTER_LAT", "center_lat")),
        "center_lon": _maybe_float(_metadata_value(row, "CENTER_LON", "center_lon")),
        "contains_point": coverage_method == "footprint_contains",
        "coverage_method": coverage_method,
        "quality_tier": quality_tier,
        "quality_score": quality_score,
        "footprint_source": source,
        "footprint_wkt": row.geometry.wkt,
        "search_note": "Selected from polygon footprint geometry; center coordinates were not used as coverage proof.",
    }


def score_nac_metadata_quality(
    *,
    coverage_method: str,
    incidence_angle: float | None,
    emission_angle: float | None,
    phase_angle: float | None,
    resolution: float | None,
) -> tuple[str, float]:
    """Rank NAC observations before ISIS processing using footprint metadata only."""
    score = 0.0

    if coverage_method == "footprint_contains":
        score += 100.0
    elif coverage_method == "footprint_intersects_radius":
        score += 50.0
    else:
        return "unusable_center_only", -100.0

    if incidence_angle is None:
        tier = "metadata_incomplete"
    elif 15.0 <= incidence_angle <= 75.0:
        tier = "preferred"
        score += 60.0 - abs(incidence_angle - 45.0)
    elif 5.0 <= incidence_angle < 15.0 or 75.0 < incidence_angle <= 90.0:
        tier = "shadow_extra"
        score += 15.0 - min(abs(incidence_angle - 15.0), abs(incidence_angle - 75.0))
    else:
        tier = "rejected_extreme_incidence"
        score -= 100.0 + abs(incidence_angle - 90.0)

    high_emission = emission_angle is not None and emission_angle > 45.0
    if emission_angle is not None:
        if emission_angle <= 45.0:
            score += 20.0 - (emission_angle / 3.0)
        else:
            score -= 30.0 + (emission_angle - 45.0)
            if tier == "preferred":
                tier = "shadow_extra"

    if phase_angle is not None:
        if 10.0 <= phase_angle <= 120.0:
            score += 15.0 - abs(phase_angle - 50.0) / 10.0
        else:
            score -= 20.0 + min(abs(phase_angle - 10.0), abs(phase_angle - 120.0)) / 5.0

    if resolution is not None:
        if resolution <= 1.5:
            score += 15.0 - min(resolution, 1.5) * 5.0
        else:
            score -= 20.0 + resolution

    return tier, float(score)


def _query_footprint_geodataframe(gdf: object, lat: float, lon: float, radius_km: float, source: str):
    import pandas as pd
    from shapely.geometry import Point, box

    lon_360 = normalize_longitude_360(lon)
    point = Point(lon_360, lat)
    contains_mask = gdf.geometry.contains(point) | gdf.geometry.touches(point)
    contains_hits = gdf[contains_mask].copy()
    contains_hits["coverage_method"] = "footprint_contains"

    if radius_km > 0:
        lat_delta = km_to_lat_degrees(radius_km)
        lon_delta = km_to_lon_degrees(radius_km, lat)
        search_geom = box(lon_360 - lon_delta, lat - lat_delta, lon_360 + lon_delta, lat + lat_delta)
        radius_mask = gdf.geometry.intersects(search_geom) & ~contains_mask
        radius_hits = gdf[radius_mask].copy()
        radius_hits["coverage_method"] = "footprint_intersects_radius"
        hits = pd.concat([contains_hits, radius_hits], ignore_index=False)
    else:
        hits = contains_hits

    records: list[dict[str, object]] = []
    for _, row in hits.iterrows():
        product_value = _metadata_value(row, "PRODUCT_ID", "product_id")
        if product_value is None:
            continue
        product_id = normalize_product_id(str(product_value))
        if not _is_nac_footprint_product(product_id, row):
            continue
        records.append(_row_to_footprint_record(row, str(row["coverage_method"]), source))
    df = pd.DataFrame(records, columns=FOOTPRINT_OUTPUT_COLUMNS)
    df = df.drop_duplicates(subset=["product_id"]) if not df.empty else df
    df.attrs["warnings"] = []
    return df


def find_lroc_nac_products_from_footprints(
    lat: float,
    lon: float,
    radius_km: float = 5.0,
    max_products: int = 20,
    *,
    footprint_cache_dir: str | Path = DEFAULT_FOOTPRINT_CACHE_DIR,
    refresh_cache: bool = False,
    verbose: bool = False,
):
    """Return LROC NAC EDR products whose polygon footprint contains/intersects the query location."""
    import geopandas as gpd

    warnings: list[str] = []
    if abs(lat) > 60.0:
        return _empty_footprint_dataframe(
            [
                "The bundled LROC NAC science mission shapefile covers 60S to 60N only. "
                "Polar/global ODE EDRNAC footprint indexing is TODO; no center-coordinate fallback was used."
            ]
        )

    try:
        shp_path = ensure_nac_equatorial_footprints(cache_dir=footprint_cache_dir, refresh_cache=refresh_cache)
        if verbose:
            print(f"[footprints] reading {shp_path}")
        gdf = gpd.read_file(shp_path)
        df = _query_footprint_geodataframe(gdf, lat, lon, radius_km, "LROC NAC_EQ_SCIENCE_MISSION_360")
    except Exception as exc:
        return _empty_footprint_dataframe([f"LROC footprint search failed: {exc}"])

    if df.empty:
        warnings.append("No NAC EDR footprints contain or intersect the requested coordinate/radius.")
    selected_ids = set(select_diverse_nac_observations(df, max_products=max_products)["product_id"]) if not df.empty else set()
    df["selected"] = df["product_id"].isin(selected_ids) if not df.empty else []
    df.attrs["warnings"] = warnings
    return df


def collection_cache_path(volume: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return cache_dir / f"{volume.lower()}_data.csv"


def read_collection_csv(
    volume: str,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh_cache: bool = False,
    timeout: int = 30,
) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = collection_cache_path(volume, cache_dir)
    if cache_path.exists() and not refresh_cache:
        return cache_path.read_text(encoding="utf-8", errors="replace")

    url = collection_csv_url(volume)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        text = response.read().decode("utf-8", errors="replace")
    cache_path.write_text(text, encoding="utf-8")
    return text


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _maybe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _get_any(row: dict[str, str], normalized_lookup: dict[str, str], candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        key = normalized_lookup.get(_normalize_column_name(candidate))
        if key is not None:
            value = row.get(key)
            if value not in (None, ""):
                return value
    return None


def _lon_in_range(lon: float, min_lon: float, max_lon: float) -> bool:
    lon = lon % 360.0
    min_lon = min_lon % 360.0
    max_lon = max_lon % 360.0
    if min_lon <= max_lon:
        return min_lon <= lon <= max_lon
    return lon >= min_lon or lon <= max_lon


def _contains_point(row: dict[str, object], lat: float, lon: float) -> bool | None:
    min_lat = _maybe_float(row.get("min_lat"))
    max_lat = _maybe_float(row.get("max_lat"))
    min_lon = _maybe_float(row.get("min_lon"))
    max_lon = _maybe_float(row.get("max_lon"))
    if None in (min_lat, max_lat, min_lon, max_lon):
        return None
    assert min_lat is not None and max_lat is not None and min_lon is not None and max_lon is not None
    return min(min_lat, max_lat) <= lat <= max(min_lat, max_lat) and _lon_in_range(lon, min_lon, max_lon)


def _parse_collection_rows(text: str, volume: str, lat: float, lon: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sample = text[:4096]
    has_header = "product" in sample.lower() or "logical_identifier" in sample.lower()

    if has_header:
        reader = csv.DictReader(io.StringIO(text))
        for raw in reader:
            line = ",".join(str(v) for v in raw.values() if v is not None)
            parsed = _parse_collection_line(line, volume, lat, lon, raw)
            if parsed is not None:
                rows.append(parsed)
    else:
        for line in text.splitlines():
            parsed = _parse_collection_line(line, volume, lat, lon, None)
            if parsed is not None:
                rows.append(parsed)
    return rows


def _parse_collection_line(
    line: str,
    volume: str,
    lat: float,
    lon: float,
    raw_row: dict[str, str] | None,
) -> dict[str, object] | None:
    upper = line.upper()
    if "/NAC/" not in upper and "NAC" not in upper:
        return None

    match = IMG_PATH_PATTERN.search(line)
    if match:
        product_id = normalize_product_id(match.group("product"))
        relative_path = match.group("relative").replace("\\", "/").removeprefix("DATA/")
        img_url = f"{PDS_ROOT}/{volume}/DATA/{relative_path}"
        search_note = "IMG path parsed from collection CSV."
    else:
        product_match = PRODUCT_PATTERN.search(line)
        if product_match is None:
            return None
        product_id = normalize_product_id(product_match.group(0))
        relative_path = None
        img_url = None
        search_note = (
            "Collection CSV exposes product ID but not IMG path or footprint; "
            "processing will resolve the IMG URL with the existing product locator."
        )
    row: dict[str, object] = {
        "product_id": product_id,
        "img_url": img_url,
        "volume": volume,
        "relative_path": relative_path,
        "spacecraft_clock_start_count": None,
        "observation_time": None,
        "incidence_angle": None,
        "emission_angle": None,
        "phase_angle": None,
        "sun_azimuth": None,
        "resolution": None,
        "min_lat": None,
        "max_lat": None,
        "min_lon": None,
        "max_lon": None,
        "footprint": None,
        "contains_point": None,
        "search_note": search_note,
    }

    if raw_row:
        normalized_lookup = {_normalize_column_name(k): k for k in raw_row}
        for output_col, candidates in ANGLE_COLUMNS.items():
            row[output_col] = _maybe_float(_get_any(raw_row, normalized_lookup, candidates))
        for output_col, candidates in FOOTPRINT_COLUMNS.items():
            row[output_col] = _maybe_float(_get_any(raw_row, normalized_lookup, candidates))
        row["spacecraft_clock_start_count"] = _get_any(
            raw_row,
            normalized_lookup,
            ("spacecraft_clock_start_count", "spacecraftclockstartcount", "sclk_start_count"),
        )
        row["observation_time"] = _get_any(
            raw_row,
            normalized_lookup,
            ("start_time", "starttime", "observation_time", "product_creation_time"),
        )
        contains = _contains_point(row, lat, lon)
        row["contains_point"] = contains
        if contains is not None:
            row["search_note"] = f"{search_note} Footprint bounds parsed from collection CSV."

    return row


def find_lroc_nac_products_for_location(
    lat: float,
    lon: float,
    radius_km: float = 5.0,
    max_products: int = 20,
    max_volume: int = 80,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    refresh_cache: bool = False,
    verbose: bool = False,
) -> object:
    """Return LROC NAC EDR products proven by footprint geometry for a location."""
    return find_lroc_nac_products_from_footprints(
        lat,
        lon,
        radius_km=radius_km,
        max_products=max_products,
        refresh_cache=refresh_cache,
        verbose=verbose,
    )


def find_lroc_nac_products_from_collections(
    lat: float,
    lon: float,
    radius_km: float = 5.0,
    max_products: int = 20,
    max_volume: int = 80,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    refresh_cache: bool = False,
    verbose: bool = False,
) -> object:
    """Legacy CSV-only candidate search. Not used for coverage decisions."""
    import pandas as pd

    del radius_km
    cache_path = Path(cache_dir)
    found: list[dict[str, object]] = []
    seen: set[str] = set()
    failures: list[str] = []

    for volume in volume_names(max_volume=max_volume):
        try:
            text = read_collection_csv(volume, cache_dir=cache_path, refresh_cache=refresh_cache)
        except Exception as exc:
            failures.append(f"{volume}: {exc}")
            if verbose:
                print(f"[warn] collection unavailable for {volume}: {exc}")
            continue

        for row in _parse_collection_rows(text, volume, lat, lon):
            product_id = str(row["product_id"])
            if product_id in seen:
                continue
            seen.add(product_id)
            found.append(row)
        selected = select_diverse_nac_observations(pd.DataFrame(found), max_products=max_products)
        if len(selected) >= max_products:
            break

    df = pd.DataFrame(found)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "product_id",
                "img_url",
                "volume",
                "relative_path",
                "spacecraft_clock_start_count",
                "observation_time",
                "incidence_angle",
                "emission_angle",
                "phase_angle",
                "sun_azimuth",
                "resolution",
                "min_lat",
                "max_lat",
                "min_lon",
                "max_lon",
                "footprint",
                "contains_point",
                "search_note",
            ]
        )
        df.attrs["warnings"] = failures
        return df

    selected_ids = set(select_diverse_nac_observations(df, max_products=max_products)["product_id"])
    df["selected"] = df["product_id"].isin(selected_ids)
    df.attrs["warnings"] = failures
    return df


def select_diverse_nac_observations(df: object, max_products: int = 20) -> object:
    """Select a deterministic, angle-diverse subset when angle metadata is available."""
    import pandas as pd

    if df.empty:
        return df.copy()

    work = df.drop_duplicates(subset=["product_id"]).copy()
    contains = work.get("contains_point")
    if contains is None:
        work["_contains_rank"] = 1
    else:
        work["_contains_rank"] = contains.map({True: 0, False: 2}).fillna(1)

    if "coverage_method" in work.columns:
        coverage_rank = {
            "footprint_contains": 0,
            "footprint_intersects_radius": 1,
            "center_only_unproven": 9,
        }
        work["_coverage_rank"] = work["coverage_method"].map(coverage_rank).fillna(5)
    else:
        work["_coverage_rank"] = 5

    if "quality_tier" in work.columns:
        tier_rank = {
            "preferred": 0,
            "shadow_extra": 1,
            "metadata_incomplete": 2,
            "rejected_extreme_incidence": 9,
            "unusable_center_only": 10,
        }
        work["_quality_rank"] = work["quality_tier"].map(tier_rank).fillna(5)
    else:
        work["_quality_rank"] = 5

    if "quality_score" not in work.columns:
        work["quality_score"] = 0.0

    for col in ("incidence_angle", "sun_azimuth", "resolution"):
        if col not in work.columns:
            work[col] = pd.NA

    if work["incidence_angle"].notna().any() or work["sun_azimuth"].notna().any():
        selected_parts = []
        remaining_slots = max_products
        for _, group in work.sort_values(["_quality_rank", "_coverage_rank"], na_position="last").groupby(["_quality_rank", "_coverage_rank"], sort=True):
            if remaining_slots <= 0:
                break
            selected_group = _select_angle_diverse_group(group, remaining_slots)
            selected_parts.append(selected_group)
            remaining_slots -= len(selected_group)
        selected = pd.concat(selected_parts) if selected_parts else work.head(0)
    else:
        selected = work.sort_values(
            ["_quality_rank", "_coverage_rank", "_contains_rank", "quality_score", "product_id"],
            ascending=[True, True, True, False, True],
            na_position="last",
        ).head(max_products)

    return selected.drop(columns=[c for c in selected.columns if c.startswith("_")], errors="ignore")


def _select_angle_diverse_group(group: object, max_products: int) -> object:
    import pandas as pd

    candidates = group.sort_values(
        ["_contains_rank", "quality_score", "resolution", "product_id"],
        ascending=[True, False, True, True],
        na_position="last",
    )
    selected_indices: list[int] = []
    while len(selected_indices) < max_products and len(selected_indices) < len(candidates):
        if not selected_indices:
            selected_indices.append(candidates.index[0])
            continue
        remaining = candidates.drop(index=selected_indices)
        selected_angles = candidates.loc[selected_indices, ["incidence_angle", "sun_azimuth"]]

        def safe_float(value: object) -> float:
            return 0.0 if pd.isna(value) else float(value)

        def diversity_score(row: pd.Series) -> tuple[float, str]:
            if pd.isna(row.get("incidence_angle")) and pd.isna(row.get("sun_azimuth")):
                return (-1.0, str(row["product_id"]))
            diffs = []
            for _, selected in selected_angles.iterrows():
                inc_diff = abs(safe_float(row.get("incidence_angle")) - safe_float(selected.get("incidence_angle")))
                sun_diff = abs(safe_float(row.get("sun_azimuth")) - safe_float(selected.get("sun_azimuth")))
                diffs.append(inc_diff + min(sun_diff, 360.0 - sun_diff))
            return (min(diffs) if diffs else 0.0, str(row["product_id"]))

        best = max(remaining.iterrows(), key=lambda item: diversity_score(item[1]))
        selected_indices.append(best[0])
    return candidates.loc[selected_indices]
