"""Local raster context sources for LOLA, GRAIL, and Diviner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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


def unavailable_context(name: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": f"No local {name} raster configured yet",
        "todo": "Add path in config/context_sources.yaml",
    }


def sample_lola_context(lat: float, lon: float, radius_km: float, config: dict[str, Any]) -> dict[str, Any]:
    return _sample_single_raster("LOLA", lat, lon, radius_km, config.get("dem_path"))


def sample_grail_context(lat: float, lon: float, radius_km: float, config: dict[str, Any]) -> dict[str, Any]:
    return _sample_single_raster("GRAIL", lat, lon, radius_km, config.get("bouguer_path"))


def sample_diviner_context(lat: float, lon: float, radius_km: float, config: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "day_temp": config.get("day_temp_path"),
        "night_temp": config.get("night_temp_path"),
    }
    configured = {k: v for k, v in paths.items() if v}
    if not configured:
        return unavailable_context("Diviner")
    results: dict[str, Any] = {"available": True, "rasters": {}}
    for key, path in configured.items():
        results["rasters"][key] = _sample_single_raster(f"Diviner {key}", lat, lon, radius_km, path)
    return results


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sample_single_raster(name: str, lat: float, lon: float, radius_km: float, raster_path: str | Path | None) -> dict[str, Any]:
    if not raster_path:
        return unavailable_context(name)
    path = Path(raster_path)
    if not path.exists():
        return {
            "available": False,
            "reason": f"Configured {name} raster does not exist: {path}",
            "todo": "Update config/context_sources.yaml with a valid local raster path",
        }

    try:
        import rasterio
        import numpy as np
        from rasterio.windows import Window
    except Exception as exc:
        return {"available": False, "reason": f"rasterio is required for local {name} sampling: {exc}"}

    try:
        with rasterio.open(path) as src:
            row, col = _latlon_to_dataset_rowcol(src, lat, lon)
            res_x = abs(src.transform.a)
            res_y = abs(src.transform.e)
            if res_x == 0 or res_y == 0:
                raise ValueError("Raster transform has zero resolution.")
            half_w = max(1, int(round((radius_km * 1000.0) / res_x)))
            half_h = max(1, int(round((radius_km * 1000.0) / res_y)))
            col_off = max(0, col - half_w)
            row_off = max(0, row - half_h)
            width = min(src.width - col_off, half_w * 2 + 1)
            height = min(src.height - row_off, half_h * 2 + 1)
            window = Window(col_off, row_off, width, height)
            arr = src.read(1, window=window, masked=True)
            values = arr.compressed()
            if values.size == 0:
                stats = _empty_stats()
            else:
                stats = {
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "valid_pixel_fraction": float(values.size / arr.size),
                }
            return {
                "available": True,
                "raster_path": str(path),
                "window": {
                    "row_start": int(row_off),
                    "row_stop": int(row_off + height),
                    "col_start": int(col_off),
                    "col_stop": int(col_off + width),
                },
                "stats": stats,
            }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"Could not sample configured {name} raster: {exc}",
            "todo": "Verify raster CRS/transform and add lunar projection handling if needed.",
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


def _latlon_to_dataset_rowcol(src: Any, lat: float, lon: float) -> tuple[int, int]:
    crs_text = src.crs.to_string() if src.crs else ""
    if "4326" in crs_text or "longlat" in crs_text.lower() or "degree" in crs_text.lower():
        row, col = src.index(lon, lat)
        return int(row), int(col)
    raise ValueError("Local raster CRS is not geographic; projected lunar sampling is not configured for this source yet.")
