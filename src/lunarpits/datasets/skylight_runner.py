"""Resumable root-level runner for skylight CNN data collection."""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from lunar_tile_pipeline.projection import normalize_lon_180, normalize_lon_360
from lunarpits.location.gather_location import gather_location_context, safe_location_label
from lunarpits.tiling.ml_tiles import get_ml_tile_for_latlon, pixel_in_tile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LROC_PIT_CATALOG_URL = (
    "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/"
    "LROLRC_2001/EXTRAS/SHAPEFILE/LUNAR_PIT_LOCATIONS/LUNAR_PIT_LOCATIONS.CSV"
)
DATASET_QUEUE_COLUMNS = [
    "queue_index",
    "split_group",
    "label",
    "label_source",
    "name",
    "lat",
    "lon_180",
    "lon_360",
    "location_id",
    "status",
    "started_at",
    "completed_at",
    "num_nac_requested",
    "num_nac_completed",
    "location_path",
    "tile_json",
    "error",
    "terrain",
    "type",
    "overhangs",
    "preferred_product_ids",
    "source_row",
]
POSITIVE_AUDIT_COLUMNS = [
    "queue_index",
    "source_row",
    "name",
    "queue_lat",
    "queue_lon_360",
    "queue_lon_180",
    "catalog_lat",
    "catalog_lon_360",
    "catalog_lon_180",
    "delta_m",
    "tile_id",
    "pixel_x",
    "pixel_y",
    "audit_status",
    "error",
]
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_AUDIT_PASSED = "passed"


@dataclass(frozen=True)
class DatasetRunConfig:
    project_root: Path = PROJECT_ROOT
    queue_csv: Path = PROJECT_ROOT / "data" / "dataset_queue.csv"
    positive_audit_csv: Path = PROJECT_ROOT / "data" / "dataset_positive_coordinate_audit.csv"
    run_summary_json: Path = PROJECT_ROOT / "data" / "dataset_run_summary.json"
    catalog_cache: Path = PROJECT_ROOT / "data" / "cache" / "lroc_pits" / "LUNAR_PIT_LOCATIONS.CSV"
    max_nac: int = 5
    radius_km: float = 5.0
    tile_size_km: float = 0.256
    pixel_resolution: float = 1.0
    negative_multiplier: int = 3
    undetermined_multiplier: int = 1
    pit_exclusion_km: float = 25.0
    dry_run: bool = False
    process_nac: bool = True
    download_context: bool = True
    force_queue_rebuild: bool = False
    retry_failed: bool = True
    refresh_cache: bool = False
    force_locations: bool = False
    verbose: bool = False
    max_rows: int | None = None
    max_positive_delta_m: float = 1.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_lroc_pit_catalog(
    cache_path: str | Path = PROJECT_ROOT / "data" / "cache" / "lroc_pits" / "LUNAR_PIT_LOCATIONS.CSV",
    *,
    refresh: bool = False,
    url: str = LROC_PIT_CATALOG_URL,
) -> pd.DataFrame:
    """Download/cache and load the official LROC pit location table.

    The archive CSV is not guaranteed to be UTF-8, so loading deliberately
    tries a small encoding cascade.
    """
    path = Path(cache_path)
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)

    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            df.attrs["source_url"] = url
            df.attrs["cache_path"] = str(path)
            df.attrs["encoding"] = encoding
            return df
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read LROC pit catalog at {path}: {last_error}")


def build_dataset_queue(config: DatasetRunConfig) -> pd.DataFrame:
    catalog = load_lroc_pit_catalog(config.catalog_cache, refresh=config.refresh_cache)
    positives = _positive_rows(catalog)
    negative_count = len(positives) * config.negative_multiplier
    undetermined_count = len(positives) * config.undetermined_multiplier
    negatives = _synthetic_rows(
        positives,
        count=negative_count,
        split_group="negative",
        label="negative_auto",
        label_source="deterministic_global_stratified_far_from_lroc_pits",
        exclusion_km=config.pit_exclusion_km,
    )
    undetermined = _synthetic_rows(
        positives,
        count=undetermined_count,
        split_group="undetermined",
        label="undetermined_auto",
        label_source="deterministic_global_stratified_hard_review_pool",
        exclusion_km=config.pit_exclusion_km * 0.5,
        offset=10_000,
    )
    queue = pd.concat([positives, negatives, undetermined], ignore_index=True)
    queue.insert(0, "queue_index", range(len(queue)))
    for column in DATASET_QUEUE_COLUMNS:
        if column not in queue.columns:
            queue[column] = ""
    queue = queue[DATASET_QUEUE_COLUMNS]
    queue["status"] = STATUS_PENDING
    queue["num_nac_requested"] = int(config.max_nac)
    queue["num_nac_completed"] = 0
    return queue


def load_or_create_queue(config: DatasetRunConfig) -> pd.DataFrame:
    if config.queue_csv.exists() and not config.force_queue_rebuild:
        queue = pd.read_csv(config.queue_csv, keep_default_na=False)
        for column in DATASET_QUEUE_COLUMNS:
            if column not in queue.columns:
                queue[column] = ""
        return queue[DATASET_QUEUE_COLUMNS]
    queue = build_dataset_queue(config)
    write_queue(queue, config.queue_csv)
    return queue


def write_queue(queue: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    queue.to_csv(tmp, index=False)
    tmp.replace(path)


def audit_positive_coordinates(queue: pd.DataFrame, config: DatasetRunConfig) -> pd.DataFrame:
    catalog = load_lroc_pit_catalog(config.catalog_cache, refresh=config.refresh_cache)
    rows: list[dict[str, Any]] = []
    positives = queue[queue["split_group"] == "positive"].copy()
    for _, row in positives.iterrows():
        audit_row = _audit_positive_row(row, catalog, config)
        rows.append(audit_row)
    audit = pd.DataFrame(rows, columns=POSITIVE_AUDIT_COLUMNS)
    config.positive_audit_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.positive_audit_csv.with_suffix(config.positive_audit_csv.suffix + ".part")
    audit.to_csv(tmp, index=False)
    tmp.replace(config.positive_audit_csv)
    failures = audit[audit["audit_status"] != STATUS_AUDIT_PASSED]
    if not failures.empty:
        preview = failures[["queue_index", "name", "audit_status", "error"]].head(10).to_dict("records")
        raise RuntimeError(
            f"Positive coordinate preflight failed for {len(failures)} row(s). "
            f"See {config.positive_audit_csv}. First failures: {preview}"
        )
    return audit


def _audit_positive_row(row: pd.Series, catalog: pd.DataFrame, config: DatasetRunConfig) -> dict[str, Any]:
    base = {column: "" for column in POSITIVE_AUDIT_COLUMNS}
    base.update(
        {
            "queue_index": row.get("queue_index", ""),
            "source_row": row.get("source_row", ""),
            "name": row.get("name", ""),
            "queue_lat": row.get("lat", ""),
            "queue_lon_360": row.get("lon_360", ""),
            "queue_lon_180": row.get("lon_180", ""),
        }
    )
    try:
        source_row = int(row["source_row"])
        if source_row < 0 or source_row >= len(catalog):
            raise ValueError(f"source_row {source_row} is outside catalog range 0..{len(catalog) - 1}")
        catalog_row = catalog.iloc[source_row]
        catalog_lat = _first_number(catalog_row, "Latitude", "Pit_Latitude")
        catalog_lon = _first_number(catalog_row, "Longitude_360", "Pit_Longitude", "Longitude")
        if catalog_lat is None or catalog_lon is None:
            raise ValueError("catalog row is missing latitude/longitude")
        queue_lat = float(row["lat"])
        queue_lon_360 = normalize_lon_360(float(row["lon_360"]))
        queue_lon_180 = normalize_lon_180(float(row["lon_180"]))
        catalog_lon_360 = normalize_lon_360(catalog_lon)
        catalog_lon_180 = normalize_lon_180(catalog_lon)
        if not (-90.0 <= queue_lat <= 90.0):
            raise ValueError(f"invalid queue latitude: {queue_lat}")
        if not (0.0 <= queue_lon_360 < 360.0):
            raise ValueError(f"invalid queue lon_360: {queue_lon_360}")
        if not math.isclose(queue_lon_180, normalize_lon_180(queue_lon_360), abs_tol=1e-9):
            raise ValueError(f"lon_180/lon_360 mismatch: {queue_lon_180} vs {queue_lon_360}")
        delta_m = _moon_distance_km(queue_lat, queue_lon_360, float(catalog_lat), catalog_lon_360) * 1000.0
        tile = get_ml_tile_for_latlon(queue_lat, queue_lon_360, tile_size_m=config.tile_size_km * 1000.0, meters_per_pixel=config.pixel_resolution)
        pixel = pixel_in_tile(queue_lat, queue_lon_360, tile_size_m=config.tile_size_km * 1000.0, meters_per_pixel=config.pixel_resolution)
        base.update(
            {
                "catalog_lat": float(catalog_lat),
                "catalog_lon_360": catalog_lon_360,
                "catalog_lon_180": catalog_lon_180,
                "delta_m": delta_m,
                "tile_id": tile.tile_id,
                "pixel_x": pixel["pixel_x"],
                "pixel_y": pixel["pixel_y"],
            }
        )
        if delta_m > config.max_positive_delta_m:
            raise ValueError(f"queue coordinate differs from catalog by {delta_m:.3f} m")
        expected_name = str(catalog_row.get("Name") or "")
        if expected_name and str(row.get("name", "")) != expected_name:
            raise ValueError(f"name mismatch: queue={row.get('name')} catalog={expected_name}")
        base["audit_status"] = STATUS_AUDIT_PASSED
    except Exception as exc:
        base["audit_status"] = "failed"
        base["error"] = str(exc)
    return base


def write_run_summary(
    queue: pd.DataFrame,
    config: DatasetRunConfig,
    *,
    last_queue_index: int | None = None,
    last_event: str = "",
) -> None:
    counts = {
        f"{split_group}:{label}:{status}": int(count)
        for (split_group, label, status), count in queue.groupby(["split_group", "label", "status"]).size().items()
    }
    payload = {
        "updated_at": utc_now(),
        "queue_csv": str(config.queue_csv),
        "positive_audit_csv": str(config.positive_audit_csv),
        "last_queue_index": last_queue_index,
        "last_event": last_event,
        "total_rows": int(len(queue)),
        "counts": counts,
    }
    config.run_summary_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.run_summary_json.with_suffix(config.run_summary_json.suffix + ".part")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(config.run_summary_json)


def validate_queue_row(row: pd.Series, audit_by_index: dict[int, dict[str, Any]], config: DatasetRunConfig) -> None:
    if str(row.get("split_group", "")) not in {"positive", "negative", "undetermined"}:
        raise ValueError(f"Invalid split_group: {row.get('split_group')}")
    if not str(row.get("label", "")).strip():
        raise ValueError("Missing label")
    lat = float(row["lat"])
    lon = float(row["lon_360"])
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Invalid latitude: {lat}")
    if not (0.0 <= normalize_lon_360(lon) < 360.0):
        raise ValueError(f"Invalid longitude: {lon}")
    tile = get_ml_tile_for_latlon(lat, lon, tile_size_m=config.tile_size_km * 1000.0, meters_per_pixel=config.pixel_resolution)
    if not tile.tile_id:
        raise ValueError("Could not compute deterministic tile ID")
    if row["split_group"] == "positive":
        queue_index = int(row["queue_index"])
        audit = audit_by_index.get(queue_index)
        if not audit or audit.get("audit_status") != STATUS_AUDIT_PASSED:
            raise ValueError(f"Positive row {queue_index} was not verified by preflight audit")


def run_dataset(
    config: DatasetRunConfig,
    *,
    gatherer: Callable[..., dict[str, Any]] = gather_location_context,
) -> pd.DataFrame:
    queue = load_or_create_queue(config)
    audit = audit_positive_coordinates(queue, config)
    audit_by_index = {int(row["queue_index"]): row for row in audit.to_dict("records")}
    write_run_summary(queue, config, last_event="preflight_passed")
    if config.dry_run:
        write_run_summary(queue, config, last_event="dry_run_complete")
        return queue
    processed = 0
    for idx, row in queue.iterrows():
        if config.max_rows is not None and processed >= config.max_rows:
            break
        status = str(row.get("status", ""))
        if status == STATUS_COMPLETE:
            continue
        if status == STATUS_FAILED and not config.retry_failed:
            continue

        try:
            validate_queue_row(row, audit_by_index, config)
            queue.at[idx, "status"] = STATUS_RUNNING
            queue.at[idx, "started_at"] = utc_now()
            queue.at[idx, "completed_at"] = ""
            queue.at[idx, "error"] = ""
            write_queue(queue, config.queue_csv)
            write_run_summary(queue, config, last_queue_index=int(row["queue_index"]), last_event="row_started")
            context = gatherer(
                lat=float(row["lat"]),
                lon=float(row["lon_360"]),
                radius_km=config.radius_km,
                max_nac=config.max_nac,
                pixel_resolution=config.pixel_resolution,
                tile_size_km=config.tile_size_km,
                process_nac=config.process_nac and not config.dry_run,
                download_context=config.download_context,
                dry_run=config.dry_run,
                force=config.force_locations,
                refresh_cache=config.refresh_cache,
                output_name=str(row["location_id"]),
                verbose=config.verbose,
                preferred_product_ids=_split_preferred_products(row.get("preferred_product_ids", "")),
            )
            location_path = Path(context.get("output_dir") or config.project_root / "data" / "locations" / str(row["location_id"]))
            if context.get("skipped_duplicate_tile"):
                queue.at[idx, "status"] = STATUS_SKIPPED
                queue.at[idx, "completed_at"] = utc_now()
                queue.at[idx, "location_path"] = str(location_path)
                queue.at[idx, "tile_json"] = str(location_path / "tile.json")
                requested_dir = context.get("requested_output_dir")
                detail = f" Requested empty folder was removed: {requested_dir}." if requested_dir else ""
                queue.at[idx, "error"] = str(context.get("warnings", ["Skipped duplicate deterministic tile."])[0]) + detail
                write_queue(queue, config.queue_csv)
                write_run_summary(queue, config, last_queue_index=int(row["queue_index"]), last_event="row_skipped_duplicate")
                processed += 1
                continue
            _finalize_location_files(location_path, row, context, config)
            validation = validate_location_completion(location_path, context, config)
            num_done = _count_completed_nac(context, location_path)
            queue.at[idx, "status"] = STATUS_COMPLETE
            queue.at[idx, "completed_at"] = utc_now()
            queue.at[idx, "num_nac_completed"] = int(num_done)
            queue.at[idx, "location_path"] = str(location_path)
            queue.at[idx, "tile_json"] = str(location_path / "tile.json")
            queue.at[idx, "error"] = validation
        except KeyboardInterrupt:
            queue.at[idx, "status"] = STATUS_FAILED
            queue.at[idx, "error"] = "Interrupted by user."
            write_queue(queue, config.queue_csv)
            write_run_summary(queue, config, last_queue_index=int(row["queue_index"]), last_event="interrupted")
            raise
        except Exception as exc:
            queue.at[idx, "status"] = STATUS_FAILED
            queue.at[idx, "completed_at"] = utc_now()
            queue.at[idx, "error"] = str(exc)
        write_queue(queue, config.queue_csv)
        write_run_summary(queue, config, last_queue_index=int(row["queue_index"]), last_event=f"row_{queue.at[idx, 'status']}")
        processed += 1
    return queue


def _positive_rows(catalog: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_row, row in catalog.iterrows():
        lat = _first_number(row, "Latitude", "Pit_Latitude")
        lon = _first_number(row, "Longitude_360", "Pit_Longitude", "Longitude")
        if lat is None or lon is None or not math.isfinite(lat) or not math.isfinite(lon):
            continue
        lon360 = normalize_lon_360(lon)
        lon180 = normalize_lon_180(lon)
        name = str(row.get("Name") or f"lroc_pit_{source_row:04d}")
        location_id = safe_location_label(f"positive_{source_row:04d}_{name}")
        rows.append(
            {
                "split_group": "positive",
                "label": "positive_skylight_candidate",
                "label_source": "lroc_lunar_pit_locations_catalog",
                "name": name,
                "lat": lat,
                "lon_180": lon180,
                "lon_360": lon360,
                "location_id": location_id,
                "terrain": row.get("Terrain", ""),
                "type": row.get("Type", ""),
                "overhangs": row.get("Overhangs?", ""),
                "preferred_product_ids": ",".join(_catalog_product_ids(row)),
                "source_row": int(source_row),
            }
        )
    return pd.DataFrame(rows)


def _catalog_product_ids(row: pd.Series) -> list[str]:
    """Return official catalog useful image IDs as normalized NAC EDR IDs.

    These IDs are ranking hints only. Footprint intersection remains the
    coverage authority, so a catalog image is used only when ODE/LROC footprint
    search confirms it touches the deterministic tile.
    """
    values: list[str] = []
    for column in ("Image 1", "Image 2", "Image 3", "Stereo IDs", "Other Useful Images"):
        value = row.get(column, "")
        if pd.isna(value):
            continue
        values.extend(_extract_catalog_product_ids(str(value)))
    seen: set[str] = set()
    ordered: list[str] = []
    for product_id in values:
        if product_id not in seen:
            ordered.append(product_id)
            seen.add(product_id)
    return ordered


def _extract_catalog_product_ids(text: str) -> list[str]:
    ids: list[str] = []
    for raw in re.findall(r"M\d+[LR](?:[EC])?", text.upper()):
        if raw.endswith(("LE", "RE")):
            ids.append(raw)
        elif raw.endswith(("LC", "RC")):
            ids.append(raw[:-1] + "E")
        else:
            ids.append(raw + "E")
    return ids


def _synthetic_rows(
    positives: pd.DataFrame,
    *,
    count: int,
    split_group: str,
    label: str,
    label_source: str,
    exclusion_km: float,
    offset: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if count <= 0:
        return pd.DataFrame(rows)
    positive_points = [(float(row.lat), float(row.lon_360)) for row in positives.itertuples()]
    index = 0
    attempts = 0
    lat_bands = [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75]
    lon_step = 17.0
    while len(rows) < count and attempts < count * 200:
        band = lat_bands[(index + offset) % len(lat_bands)]
        lat = band + (((index * 37 + offset) % 1200) / 1200.0 - 0.5) * 10.0
        lat = max(-82.0, min(82.0, lat))
        lon360 = normalize_lon_360((index * lon_step + offset * 0.013) % 360.0)
        attempts += 1
        index += 1
        if _near_any_positive(lat, lon360, positive_points, exclusion_km):
            continue
        lon180 = normalize_lon_180(lon360)
        name = f"{split_group}_{len(rows):04d}"
        rows.append(
            {
                "split_group": split_group,
                "label": label,
                "label_source": label_source,
                "name": name,
                "lat": lat,
                "lon_180": lon180,
                "lon_360": lon360,
                "location_id": safe_location_label(f"{split_group}_{len(rows):04d}_{lat:.6f}_{lon360:.6f}"),
                "terrain": "auto_stratified",
                "type": "",
                "overhangs": "",
                "preferred_product_ids": "",
                "source_row": "",
            }
        )
    if len(rows) < count:
        raise RuntimeError(f"Could only generate {len(rows)} {split_group} rows out of requested {count}.")
    return pd.DataFrame(rows)


def _near_any_positive(lat: float, lon360: float, positives: list[tuple[float, float]], exclusion_km: float) -> bool:
    return any(_moon_distance_km(lat, lon360, p_lat, p_lon) < exclusion_km for p_lat, p_lon in positives)


def _moon_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 1737.4
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlon = math.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlon / 2.0) ** 2
    return radius_km * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _first_number(row: pd.Series, *names: str) -> float | None:
    for name in names:
        if name in row and not pd.isna(row[name]):
            try:
                return float(row[name])
            except (TypeError, ValueError):
                continue
    return None


def _finalize_location_files(location_path: Path, row: pd.Series, context: dict[str, Any], config: DatasetRunConfig) -> None:
    location_path.mkdir(parents=True, exist_ok=True)
    annotations = _annotations_payload(row)
    annotations_path = location_path / "annotations.csv"
    pd.DataFrame([annotations]).to_csv(annotations_path, index=False)

    mask_readme = location_path / "nac" / "masks" / "README.md"
    mask_readme.parent.mkdir(parents=True, exist_ok=True)
    if not mask_readme.exists():
        mask_readme.write_text(
            "Positive skylight candidates need manual instance masks here. "
            "Negative and undetermined auto rows do not require masks for initial collection.\n",
            encoding="utf-8",
        )

    tile_json = location_path / "tile.json"
    tile_payload = _read_json(tile_json)
    if not tile_payload:
        tile_payload = context.get("tile", {})
    tile_payload["dataset_label"] = annotations
    tile_payload["dataset_queue"] = {
        "queue_csv": str(config.queue_csv),
        "queue_index": int(row["queue_index"]),
        "split_group": row["split_group"],
        "status": STATUS_COMPLETE,
    }
    tile_payload["context"] = {
        "gravity": context.get("gravity", {}),
        "topology": context.get("topology", context.get("lola", {})),
        "ir": context.get("ir", context.get("diviner", {})),
    }
    _write_json(tile_json, tile_payload)

    manifest = _manifest_rows(location_path, row, context, annotations_path)
    manifest_csv = location_path / "manifest.csv"
    manifest_parquet = location_path / "manifest.parquet"
    manifest.to_csv(manifest_csv, index=False)
    try:
        manifest.to_parquet(manifest_parquet, index=False)
    except Exception:
        pass


def _annotations_payload(row: pd.Series) -> dict[str, Any]:
    return {
        "location_id": row["location_id"],
        "split_group": row["split_group"],
        "label": row["label"],
        "label_source": row["label_source"],
        "name": row["name"],
        "lat": float(row["lat"]),
        "lon_180": float(row["lon_180"]),
        "lon_360": float(row["lon_360"]),
        "mask_status": "needs_manual_mask" if row["split_group"] == "positive" else "not_required",
        "terrain": row.get("terrain", ""),
        "type": row.get("type", ""),
        "overhangs": row.get("overhangs", ""),
        "preferred_product_ids": row.get("preferred_product_ids", ""),
        "notes": "",
    }


def _manifest_rows(location_path: Path, row: pd.Series, context: dict[str, Any], annotations_path: Path) -> pd.DataFrame:
    processed = context.get("lroc_nac", {}).get("processed", [])
    rows: list[dict[str, Any]] = []
    if not processed:
        rows.append(_manifest_row(location_path, row, context, annotations_path, {}))
    else:
        for product in processed:
            rows.append(_manifest_row(location_path, row, context, annotations_path, product))
    return pd.DataFrame(rows)


def validate_location_completion(location_path: Path, context: dict[str, Any], config: DatasetRunConfig) -> str:
    required = [
        location_path / "tile.json",
        location_path / "manifest.csv",
        location_path / "annotations.csv",
        location_path / "nac" / "masks" / "README.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Completion validation failed; missing required files: {missing}")
    manifest = pd.read_csv(location_path / "manifest.csv", keep_default_na=False)
    if manifest.empty:
        raise RuntimeError("Completion validation failed; manifest.csv is empty")
    valid_images = 0
    invalid_reasons: list[str] = []
    for _, row in manifest.iterrows():
        image_path = Path(str(row.get("image_path", "")))
        metadata_path = Path(str(row.get("metadata_path", "")))
        if not image_path.exists():
            invalid_reasons.append(f"missing image {image_path}")
            continue
        if not metadata_path.exists():
            invalid_reasons.append(f"missing sidecar {metadata_path}")
            continue
        sidecar = _read_json(metadata_path)
        valid_fraction = float(sidecar.get("valid_pixel_fraction") or 0.0)
        if valid_fraction < 0.80:
            invalid_reasons.append(f"{image_path.name} valid fraction {valid_fraction:.3f}")
            continue
        valid_images += 1
    if config.process_nac and valid_images < 1:
        raise RuntimeError(f"Completion validation failed; no valid NAC image tiles. Reasons: {invalid_reasons[:5]}")
    requested = int(config.max_nac)
    if valid_images < requested:
        return f"Completed with {valid_images}/{requested} valid NAC tiles; rejected/failed products are recorded in audit.json."
    return ""


def _manifest_row(
    location_path: Path,
    row: pd.Series,
    context: dict[str, Any],
    annotations_path: Path,
    product: dict[str, Any],
) -> dict[str, Any]:
    return {
        "location_id": row["location_id"],
        "split_group": row["split_group"],
        "label": row["label"],
        "lat": float(row["lat"]),
        "lon_180": float(row["lon_180"]),
        "lon_360": float(row["lon_360"]),
        "tile_id": context.get("tile", {}).get("tile_id", ""),
        "product_id": product.get("product_id", ""),
        "image_path": product.get("context_tif") or product.get("crop_tif") or "",
        "preview_path": product.get("quicklook") or product.get("crop_quicklook") or "",
        "metadata_path": _product_metadata_path(location_path, product),
        "tile_json": str(location_path / "tile.json"),
        "annotations_csv": str(annotations_path),
        "mask_status": "needs_manual_mask" if row["split_group"] == "positive" else "not_required",
        "gravity_available": bool(context.get("gravity", {}).get("available")),
        "topology_available": bool(context.get("topology", {}).get("available")),
        "ir_available": bool(context.get("ir", {}).get("available")),
    }


def _product_metadata_path(location_path: Path, product: dict[str, Any]) -> str:
    image_path = product.get("context_tif") or product.get("crop_tif") or ""
    if image_path:
        return str(Path(image_path).with_suffix(".json"))
    product_id = product.get("product_id")
    if product_id:
        return str(location_path / "nac" / "images" / f"{product_id}.json")
    return ""


def _count_completed_nac(context: dict[str, Any], location_path: Path) -> int:
    processed = context.get("lroc_nac", {}).get("processed", [])
    if processed:
        return sum(1 for item in processed if item.get("context_tif") or item.get("crop_tif"))
    images = location_path / "nac" / "images"
    return len([path for path in images.glob("*.tif") if path.is_file()]) if images.exists() else 0


def _split_preferred_products(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item.strip().upper() for item in text.split(",") if item.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a resumable skylight CNN coordinate dataset.")
    parser.add_argument("--queue-csv", default=str(PROJECT_ROOT / "data" / "dataset_queue.csv"))
    parser.add_argument("--positive-audit-csv", default=str(PROJECT_ROOT / "data" / "dataset_positive_coordinate_audit.csv"))
    parser.add_argument("--run-summary-json", default=str(PROJECT_ROOT / "data" / "dataset_run_summary.json"))
    parser.add_argument("--max-nac", type=int, default=5)
    parser.add_argument("--radius-km", type=float, default=5.0)
    parser.add_argument("--tile-size-km", type=float, default=0.256)
    parser.add_argument("--pixel-resolution", type=float, default=1.0)
    parser.add_argument("--negative-multiplier", type=int, default=3)
    parser.add_argument("--undetermined-multiplier", type=int, default=1)
    parser.add_argument("--pit-exclusion-km", type=float, default=25.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-process-nac", dest="process_nac", action="store_false", default=True)
    parser.add_argument("--no-download-context", dest="download_context", action="store_false", default=True)
    parser.add_argument("--force-queue-rebuild", action="store_true")
    parser.add_argument("--no-retry-failed", dest="retry_failed", action="store_false", default=True)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--force-locations", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None, help="Process at most this many unfinished rows in this run.")
    parser.add_argument("--max-positive-delta-m", type=float, default=1.0)
    parser.add_argument("--print-summary", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> DatasetRunConfig:
    return DatasetRunConfig(
        queue_csv=Path(args.queue_csv),
        positive_audit_csv=Path(args.positive_audit_csv),
        run_summary_json=Path(args.run_summary_json),
        max_nac=args.max_nac,
        radius_km=args.radius_km,
        tile_size_km=args.tile_size_km,
        pixel_resolution=args.pixel_resolution,
        negative_multiplier=args.negative_multiplier,
        undetermined_multiplier=args.undetermined_multiplier,
        pit_exclusion_km=args.pit_exclusion_km,
        dry_run=args.dry_run,
        process_nac=args.process_nac,
        download_context=args.download_context,
        force_queue_rebuild=args.force_queue_rebuild,
        retry_failed=args.retry_failed,
        refresh_cache=args.refresh_cache,
        force_locations=args.force_locations,
        verbose=args.verbose,
        max_rows=args.max_rows,
        max_positive_delta_m=args.max_positive_delta_m,
    )


def print_summary(queue: pd.DataFrame) -> None:
    print("Queue summary")
    print(queue.groupby(["split_group", "label", "status"]).size().to_string())
    print(f"\nqueue rows: {len(queue)}")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = config_from_args(args)
    queue = run_dataset(config)
    print_summary(queue)
    print(f"\nqueue csv: {config.queue_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
