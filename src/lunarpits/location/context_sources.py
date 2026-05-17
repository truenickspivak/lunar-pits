"""Tile-based context samplers for LOLA, GRAIL, and Diviner.

The first implementation deliberately favors small/coarse global products over
large multi-GB rasters. The output JSON always records the native resolution so
the user can see when a 0.5 km tile is being represented by coarser context.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from lunar_tile_pipeline.projection import normalize_lon_360
from lunar_tile_pipeline.tiling import LunarTile, get_tile_for_latlon


DEFAULT_CACHE_DIR = Path("data/cache/context_sources")
LOLA_LDEM16_SHAPE = (2880, 5760)
LOLA_LDEM16_PIXELS_PER_DEGREE = 16.0
LOLA_LDEM16_SCALE_M = 0.5


def load_context_config(path: str | Path = "config/context_sources.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    try:
        import yaml  # type: ignore

        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return _load_minimal_yaml(config_path.read_text(encoding="utf-8"))


def _load_minimal_yaml(text: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current = {}
            config[line[:-1].strip()] = current
            continue
        if current is not None and ":" in line:
            key, value = line.strip().split(":", 1)
            value = value.strip()
            current[key.strip()] = None if value in {"", "null", "None"} else value.strip("'\"")
    return config


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def unavailable_context(name: str, reason: str | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "available": False,
        "dataset": name,
        "reason": reason or f"No local or cached {name} source configured yet",
        "todo": "Add a local path in config/context_sources.yaml or rerun with --download-context.",
    }
    if config:
        for key in ("dem_url", "bouguer_url", "max_temp_anomaly_url", "min_temp_anomaly_url", "high_res_dem_url"):
            if config.get(key):
                payload[key] = config[key]
    return payload


def sample_lola_context_for_tile(
    tile: LunarTile,
    config: dict[str, Any],
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    download: bool = False,
) -> dict[str, Any]:
    """Sample the coarse LOLA LDEM_16 PDS IMG product for a deterministic tile."""
    cache = Path(cache_dir) / "lola"
    source = _resolve_source(
        "LOLA",
        config,
        path_key="dem_path",
        url_key="dem_url",
        cache_name_key="cache_name",
        cache_dir=cache,
        download=download,
    )
    label = _resolve_source(
        "LOLA label",
        config,
        path_key="label_path",
        url_key="label_url",
        cache_name_key="label_cache_name",
        cache_dir=cache,
        download=download,
        required=False,
    )
    if not source["available"]:
        source.update(
            {
                "dataset": "LOLA",
                "native_resolution_m_per_pixel_estimate": config.get("native_resolution_m_per_pixel", 1895.21),
                "tile_pixels_across_native_estimate": tile.tile_size_m / float(config.get("native_resolution_m_per_pixel", 1895.21)),
                "note": "The default LOLA context uses the small LDEM_16 grid; the 118 m GeoTIFF is recorded as high_res_dem_url but not downloaded automatically.",
            }
        )
        return source

    try:
        stats_payload = _sample_lola_ldem16(Path(source["path"]), tile)
        stats_payload.update(
            {
                "available": True,
                "dataset": "LOLA",
                "source_file": source["path"],
                "source_url": config.get("dem_url"),
                "label_file": label.get("path") if label.get("available") else None,
                "high_res_dem_url": config.get("high_res_dem_url"),
                "native_resolution_m_per_pixel_estimate": config.get("native_resolution_m_per_pixel", 1895.21),
                "tile_pixels_across_native_estimate": tile.tile_size_m / float(config.get("native_resolution_m_per_pixel", 1895.21)),
                "tile_id": tile.tile_id,
                "tile_size_km": tile.tile_size_km,
            }
        )
        return stats_payload
    except Exception as exc:
        return unavailable_context("LOLA", f"Could not sample LOLA LDEM_16: {exc}", config)


def sample_grail_context_for_tile(
    tile: LunarTile,
    config: dict[str, Any],
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    download: bool = False,
) -> dict[str, Any]:
    source = _resolve_source(
        "GRAIL",
        config,
        path_key="bouguer_path",
        url_key="bouguer_url",
        cache_name_key="cache_name",
        cache_dir=Path(cache_dir) / "grail",
        download=download,
    )
    if not source["available"]:
        source.update(
            {
                "dataset": "GRAIL",
                "native_resolution_m_per_pixel_estimate": config.get("native_resolution_m_per_pixel_estimate"),
                "tile_pixels_across_native_estimate": _tile_pixels_estimate(tile, config),
            }
        )
        return source
    return _sample_geographic_raster("GRAIL", Path(source["path"]), tile, config.get("bouguer_url"), config)


def sample_diviner_context_for_tile(
    tile: LunarTile,
    config: dict[str, Any],
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    download: bool = False,
) -> dict[str, Any]:
    products = {
        "max_temp_anomaly": ("max_temp_anomaly_path", "max_temp_anomaly_url", "max_temp_anomaly_cache_name"),
        "min_temp_anomaly": ("min_temp_anomaly_path", "min_temp_anomaly_url", "min_temp_anomaly_cache_name"),
    }
    rasters: dict[str, Any] = {}
    any_available = False
    for name, (path_key, url_key, cache_key) in products.items():
        source = _resolve_source(
            f"Diviner {name}",
            config,
            path_key=path_key,
            url_key=url_key,
            cache_name_key=cache_key,
            cache_dir=Path(cache_dir) / "diviner",
            download=download,
        )
        if source["available"]:
            any_available = True
            rasters[name] = _sample_diviner_xyz(Path(source["path"]), tile, source_url=config.get(url_key), config=config)
        else:
            rasters[name] = source
    return {
        "available": any_available,
        "dataset": "Diviner",
        "tile_id": tile.tile_id,
        "tile_size_km": tile.tile_size_km,
        "native_resolution_m_per_pixel_estimate": config.get("native_resolution_m_per_pixel_estimate"),
        "tile_pixels_across_native_estimate": _tile_pixels_estimate(tile, config),
        "rasters": rasters,
    }


def sample_lola_context(lat: float, lon: float, radius_km: float, config: dict[str, Any]) -> dict[str, Any]:
    tile = get_tile_for_latlon(lat, lon, tile_size_km=max(radius_km * 2.0, 0.5))
    return sample_lola_context_for_tile(tile, config, download=False)


def sample_grail_context(lat: float, lon: float, radius_km: float, config: dict[str, Any]) -> dict[str, Any]:
    tile = get_tile_for_latlon(lat, lon, tile_size_km=max(radius_km * 2.0, 0.5))
    return sample_grail_context_for_tile(tile, config, download=False)


def sample_diviner_context(lat: float, lon: float, radius_km: float, config: dict[str, Any]) -> dict[str, Any]:
    tile = get_tile_for_latlon(lat, lon, tile_size_km=max(radius_km * 2.0, 0.5))
    return sample_diviner_context_for_tile(tile, config, download=False)


def _resolve_source(
    name: str,
    config: dict[str, Any],
    *,
    path_key: str,
    url_key: str,
    cache_name_key: str,
    cache_dir: Path,
    download: bool,
    required: bool = True,
) -> dict[str, Any]:
    configured_path = config.get(path_key)
    if configured_path and Path(configured_path).exists():
        return {"available": True, "path": str(Path(configured_path)), "source": "configured_path"}

    url = config.get(url_key)
    cache_name = config.get(cache_name_key) or (Path(str(url)).name if url else None)
    cached = cache_dir / str(cache_name) if cache_name else None
    if cached and cached.exists():
        return {"available": True, "path": str(cached), "source": "cache", "url": url}

    if url and cached and download:
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            _download_file(str(url), cached)
            return {"available": True, "path": str(cached), "source": "download", "url": url}
        except Exception as exc:
            return {
                "available": False,
                "dataset": name,
                "reason": f"Could not download {name} source: {exc}",
                "url": url,
                "cache_path": str(cached),
                "todo": "Download this source manually to the cache path or configure a local path in config/context_sources.yaml.",
            }

    if not required:
        return {"available": False, "reason": f"Optional {name} source is not cached.", "url": url}
    return {
        "available": False,
        "dataset": name,
        "reason": f"{name} source is not cached and download was not requested.",
        "url": url,
        "cache_path": str(cached) if cached else None,
        "todo": "Rerun with --download-context to cache this coarse source, or set a local path in config/context_sources.yaml.",
    }


def _download_file(url: str, out_path: Path) -> None:
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        curl = shutil.which("curl.exe") or shutil.which("curl")
        if not curl:
            raise
        completed = subprocess.run(
            [curl, "-L", "--ssl-no-revoke", "-o", str(tmp_path), url],
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError((completed.stderr or completed.stdout or "curl download failed").strip())
    tmp_path.replace(out_path)


def _sample_lola_ldem16(path: Path, tile: LunarTile) -> dict[str, Any]:
    import numpy as np

    rows, cols = LOLA_LDEM16_SHAPE
    raster = np.memmap(path, dtype="<i2", mode="r", shape=(rows, cols))
    row_center, col_center = _ldem16_row_col(tile.center_lat, tile.center_lon_360)
    corner_pairs = [_ldem16_row_col(corner["lat"], corner["lon_360"]) for corner in tile.bounds_latlon["corner_latlon"]]  # type: ignore[index]
    row_values = [row for row, _ in corner_pairs] + [row_center]
    col_values = [col for _, col in corner_pairs] + [col_center]
    row_start = max(0, min(row_values) - 1)
    row_stop = min(rows, max(row_values) + 2)
    col_start = max(0, min(col_values) - 1)
    col_stop = min(cols, max(col_values) + 2)
    arr = np.asarray(raster[row_start:row_stop, col_start:col_stop], dtype=np.float64) * LOLA_LDEM16_SCALE_M
    values = arr[np.isfinite(arr)]
    nearest_m = float(raster[row_center, col_center]) * LOLA_LDEM16_SCALE_M
    return {
        "sample_mode": "window_with_nearest_center",
        "window": {
            "row_start": int(row_start),
            "row_stop": int(row_stop),
            "col_start": int(col_start),
            "col_stop": int(col_stop),
            "center_row": int(row_center),
            "center_col": int(col_center),
        },
        "center_value_m": nearest_m,
        "stats": _stats(values, arr.size),
        "note": "LOLA LDEM_16 is much coarser than a 0.5 km tile; center_value_m is the nearest coarse elevation sample.",
    }


def _ldem16_row_col(lat: float, lon360: float) -> tuple[int, int]:
    row = int(math.floor((90.0 - float(lat)) * LOLA_LDEM16_PIXELS_PER_DEGREE))
    col = int(math.floor(normalize_lon_360(lon360) * LOLA_LDEM16_PIXELS_PER_DEGREE))
    return max(0, min(LOLA_LDEM16_SHAPE[0] - 1, row)), max(0, min(LOLA_LDEM16_SHAPE[1] - 1, col))


def _sample_geographic_raster(name: str, path: Path, tile: LunarTile, source_url: str | None, config: dict[str, Any]) -> dict[str, Any]:
    try:
        import numpy as np
        import rasterio
        from rasterio.windows import Window
    except Exception as exc:
        return unavailable_context(name, f"rasterio/numpy is required for {name} sampling: {exc}", config)

    try:
        with rasterio.open(path) as src:
            row_center, col_center = _dataset_row_col(src, tile.center_lat, tile.center_lon_360)
            row_cols = [_dataset_row_col(src, corner["lat"], corner["lon_360"]) for corner in tile.bounds_latlon["corner_latlon"]]  # type: ignore[index]
            rows = [row for row, _ in row_cols] + [row_center]
            cols = [col for _, col in row_cols] + [col_center]
            row_start = max(0, min(rows) - 1)
            row_stop = min(src.height, max(rows) + 2)
            col_start = max(0, min(cols) - 1)
            col_stop = min(src.width, max(cols) + 2)
            window = Window(col_start, row_start, max(1, col_stop - col_start), max(1, row_stop - row_start))
            arr = src.read(1, window=window, masked=True)
            values = arr.compressed()
            center_value = src.read(1, window=Window(col_center, row_center, 1, 1), masked=True)
            center = center_value.compressed()
            return {
                "available": True,
                "dataset": name,
                "source_file": str(path),
                "source_url": source_url,
                "tile_id": tile.tile_id,
                "tile_size_km": tile.tile_size_km,
                "native_resolution": {"x": abs(src.transform.a), "y": abs(src.transform.e), "units": "dataset_units"},
                "native_resolution_m_per_pixel_estimate": config.get("native_resolution_m_per_pixel_estimate"),
                "tile_pixels_across_native_estimate": _tile_pixels_estimate(tile, config),
                "sample_mode": "window_with_nearest_center",
                "center_value": float(center[0]) if center.size else None,
                "window": {
                    "row_start": int(row_start),
                    "row_stop": int(row_stop),
                    "col_start": int(col_start),
                    "col_stop": int(col_stop),
                    "center_row": int(row_center),
                    "center_col": int(col_center),
                },
                "stats": _stats(values, arr.size),
            }
    except Exception as exc:
        return unavailable_context(name, f"Could not sample {name} raster: {exc}", config)


def _dataset_row_col(src: Any, lat: float, lon360: float) -> tuple[int, int]:
    candidates = [normalize_lon_360(lon360), ((normalize_lon_360(lon360) + 180.0) % 360.0) - 180.0]
    last: Exception | None = None
    for lon in candidates:
        try:
            row, col = src.index(lon, lat)
            if 0 <= row < src.height and 0 <= col < src.width:
                return int(row), int(col)
        except Exception as exc:
            last = exc
    if last:
        raise last
    row, col = src.index(candidates[0], lat)
    return int(row), int(col)


def _sample_diviner_xyz(path: Path, tile: LunarTile, *, source_url: str | None, config: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        return unavailable_context("Diviner", f"XYZ file has fewer than 3 columns: {path}", config)

    lon = np.mod(data[:, 0], 360.0)
    lat = data[:, 1]
    values = data[:, 2]
    bounds = tile.bounds_latlon
    min_lat = float(bounds["min_lat_approx"])
    max_lat = float(bounds["max_lat_approx"])
    lon_min = float(bounds["min_lon_360_approx"])
    lon_max = float(bounds["max_lon_360_approx"])
    if bounds.get("crosses_lon_360_wrap"):
        lon_mask = (lon >= lon_min) | (lon <= lon_max)
    else:
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
    mask = lon_mask & (lat >= min_lat) & (lat <= max_lat) & np.isfinite(values)

    if np.any(mask):
        sample_values = values[mask]
        sample_mode = "points_inside_tile_bounds"
        nearest_payload: dict[str, Any] = {}
    else:
        dlon = ((lon - tile.center_lon_360 + 180.0) % 360.0) - 180.0
        dlat = lat - tile.center_lat
        distance_deg = np.sqrt((dlon * math.cos(math.radians(tile.center_lat))) ** 2 + dlat**2)
        nearest_idx = int(np.nanargmin(distance_deg))
        sample_values = np.asarray([values[nearest_idx]])
        sample_mode = "nearest_point"
        nearest_payload = {
            "nearest_lat": float(lat[nearest_idx]),
            "nearest_lon_360": float(lon[nearest_idx]),
            "nearest_distance_degrees_approx": float(distance_deg[nearest_idx]),
        }

    return {
        "available": True,
        "dataset": "Diviner",
        "source_file": str(path),
        "source_url": source_url,
        "tile_id": tile.tile_id,
        "tile_size_km": tile.tile_size_km,
        "native_resolution_m_per_pixel_estimate": config.get("native_resolution_m_per_pixel_estimate"),
        "tile_pixels_across_native_estimate": _tile_pixels_estimate(tile, config),
        "sample_mode": sample_mode,
        "num_points": int(sample_values.size),
        "stats": _stats(sample_values, max(1, int(sample_values.size))),
        **nearest_payload,
        "note": "Diviner anomaly grids are very coarse compared with a 0.5 km tile; nearest_point is expected for many tiles.",
    }


def _stats(values: Any, total_size: int) -> dict[str, float]:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return _empty_stats()
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "valid_pixel_fraction": float(arr.size / max(1, total_size)),
    }


def _empty_stats() -> dict[str, float]:
    return {
        "min": 0.0,
        "max": 0.0,
        "mean": 0.0,
        "median": 0.0,
        "std": 0.0,
        "valid_pixel_fraction": 0.0,
    }


def _tile_pixels_estimate(tile: LunarTile, config: dict[str, Any]) -> float | None:
    resolution = config.get("native_resolution_m_per_pixel") or config.get("native_resolution_m_per_pixel_estimate")
    if not resolution:
        return None
    return float(tile.tile_size_m) / float(resolution)
