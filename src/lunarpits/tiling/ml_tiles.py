"""Production lunar ML tile grid and fixed radiometric preview policy.

This module is intentionally separate from the older exploratory
``lunar_tile_pipeline`` grid.  The production ML grid uses tile indices
anchored at global projected x/y multiples so the same lunar coordinate always
lands in the same tile and pixel offset.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

MOON_RADIUS_M = 1737400.0
DEFAULT_TILE_SIZE_M = 256.0
DEFAULT_METERS_PER_PIXEL = 1.0
DEFAULT_NODATA_VALUE = -9999.0
DEFAULT_PREVIEW_MIN = 0.0
DEFAULT_PREVIEW_MAX = 0.12
PRODUCTION_NORMALIZATION_POLICIES = {
    "preserve_float32",
    "fixed_global_clip",
    "dataset_percentile_constants",
}
FORBIDDEN_PRODUCTION_NAME_TOKENS = (
    "browse",
    "enhanced",
    "stretch",
    "equalized",
    "clahe",
    "sharpen",
    "gamma",
    "pretty",
    "quicklook",
)


@dataclass(frozen=True)
class MlTileSpec:
    """A deterministic production ML tile cell."""

    tile_i: int
    tile_j: int
    tile_size_m: float
    meters_per_pixel: float
    tile_size_px: int
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    ul_x: float
    ul_y: float
    center_x_m: float
    center_y_m: float
    center_lat: float
    center_lon_180: float
    center_lon_360: float
    tile_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MlScalingPolicy:
    """Radiometric policy shared by ML arrays and human previews."""

    normalization_policy: str = "preserve_float32"
    preview_min: float = DEFAULT_PREVIEW_MIN
    preview_max: float = DEFAULT_PREVIEW_MAX
    ml_clip_min: float | None = None
    ml_clip_max: float | None = None
    nodata_value: float = DEFAULT_NODATA_VALUE

    def __post_init__(self) -> None:
        if self.normalization_policy not in PRODUCTION_NORMALIZATION_POLICIES:
            raise ValueError(f"Forbidden or unknown production normalization policy: {self.normalization_policy}")
        if not self.preview_max > self.preview_min:
            raise ValueError("preview_max must be greater than preview_min.")
        if self.normalization_policy in {"fixed_global_clip", "dataset_percentile_constants"}:
            if self.ml_clip_min is None or self.ml_clip_max is None:
                raise ValueError(f"{self.normalization_policy} requires ml_clip_min and ml_clip_max.")
            if not self.ml_clip_max > self.ml_clip_min:
                raise ValueError("ml_clip_max must be greater than ml_clip_min.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_lon_180(lon: float) -> float:
    """Normalize longitude to [-180, 180)."""
    return ((float(lon) + 180.0) % 360.0) - 180.0


def normalize_lon_360(lon: float) -> float:
    """Normalize longitude to [0, 360)."""
    return float(lon) % 360.0


def latlon_to_global_xy(
    lat: float,
    lon: float,
    *,
    radius_m: float = MOON_RADIUS_M,
    center_lat: float = 0.0,
    center_lon: float = 0.0,
) -> tuple[float, float]:
    """Convert lunar lat/lon to first-version global equirectangular meters."""
    lon_delta = normalize_lon_180(float(lon) - float(center_lon))
    x = radius_m * math.radians(lon_delta) * math.cos(math.radians(center_lat))
    y = radius_m * math.radians(float(lat) - float(center_lat))
    return x, y


def global_xy_to_latlon(
    x: float,
    y: float,
    *,
    radius_m: float = MOON_RADIUS_M,
    center_lat: float = 0.0,
    center_lon: float = 0.0,
) -> tuple[float, float]:
    """Inverse first-version global equirectangular projection."""
    lat = math.degrees(float(y) / radius_m) + float(center_lat)
    lon = math.degrees(float(x) / (radius_m * math.cos(math.radians(center_lat)))) + float(center_lon)
    return lat, normalize_lon_180(lon)


def assert_integer_tile_pixels(tile_size_m: float, meters_per_pixel: float) -> int:
    """Return tile size in pixels, rejecting non-integer production grids."""
    if tile_size_m <= 0:
        raise ValueError("tile_size_m must be positive.")
    if meters_per_pixel <= 0:
        raise ValueError("meters_per_pixel must be positive.")
    exact = float(tile_size_m) / float(meters_per_pixel)
    rounded = round(exact)
    if not math.isclose(exact, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("tile_size_m / meters_per_pixel must be an integer.")
    return int(rounded)


def tile_indices_for_xy(x: float, y: float, tile_size_m: float = DEFAULT_TILE_SIZE_M) -> tuple[int, int]:
    """Map projected meters to deterministic global production tile indices."""
    if tile_size_m <= 0:
        raise ValueError("tile_size_m must be positive.")
    return math.floor(float(x) / tile_size_m), math.floor(float(y) / tile_size_m)


def make_ml_tile_id(tile_i: int, tile_j: int) -> str:
    return f"tile_x{tile_i:+07d}_y{tile_j:+07d}"


def tile_bounds_xy(tile_i: int, tile_j: int, tile_size_m: float = DEFAULT_TILE_SIZE_M) -> tuple[float, float, float, float]:
    x_min = int(tile_i) * float(tile_size_m)
    y_min = int(tile_j) * float(tile_size_m)
    return x_min, y_min, x_min + float(tile_size_m), y_min + float(tile_size_m)


def get_ml_tile_for_xy(
    x: float,
    y: float,
    *,
    tile_size_m: float = DEFAULT_TILE_SIZE_M,
    meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL,
) -> MlTileSpec:
    tile_i, tile_j = tile_indices_for_xy(x, y, tile_size_m=tile_size_m)
    tile_size_px = assert_integer_tile_pixels(tile_size_m, meters_per_pixel)
    x_min, y_min, x_max, y_max = tile_bounds_xy(tile_i, tile_j, tile_size_m=tile_size_m)
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    center_lat, center_lon_180 = global_xy_to_latlon(center_x, center_y)
    return MlTileSpec(
        tile_i=tile_i,
        tile_j=tile_j,
        tile_size_m=float(tile_size_m),
        meters_per_pixel=float(meters_per_pixel),
        tile_size_px=tile_size_px,
        x_min_m=x_min,
        x_max_m=x_max,
        y_min_m=y_min,
        y_max_m=y_max,
        ul_x=x_min,
        ul_y=y_max,
        center_x_m=center_x,
        center_y_m=center_y,
        center_lat=center_lat,
        center_lon_180=center_lon_180,
        center_lon_360=normalize_lon_360(center_lon_180),
        tile_id=make_ml_tile_id(tile_i, tile_j),
    )


def get_ml_tile_for_latlon(
    lat: float,
    lon: float,
    *,
    tile_size_m: float = DEFAULT_TILE_SIZE_M,
    meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL,
) -> MlTileSpec:
    x, y = latlon_to_global_xy(lat, lon)
    return get_ml_tile_for_xy(x, y, tile_size_m=tile_size_m, meters_per_pixel=meters_per_pixel)


def pixel_in_tile(
    lat: float,
    lon: float,
    *,
    tile_size_m: float = DEFAULT_TILE_SIZE_M,
    meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL,
) -> dict[str, float | int]:
    """Return deterministic tile and pixel offset for a coordinate."""
    x, y = latlon_to_global_xy(lat, lon)
    tile = get_ml_tile_for_xy(x, y, tile_size_m=tile_size_m, meters_per_pixel=meters_per_pixel)
    pixel_x = math.floor((x - tile.x_min_m) / meters_per_pixel)
    pixel_y = math.floor((tile.y_max_m - y) / meters_per_pixel)
    return {
        "x_m": x,
        "y_m": y,
        "tile_i": tile.tile_i,
        "tile_j": tile.tile_j,
        "tile_id": tile.tile_id,
        "pixel_x": int(pixel_x),
        "pixel_y": int(pixel_y),
    }


def snap_bounds_to_tile_grid(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    tile_size_m: float = DEFAULT_TILE_SIZE_M,
) -> tuple[float, float, float, float]:
    """Expand projected bounds to the production tile lattice."""
    return (
        math.floor(float(min_x) / tile_size_m) * tile_size_m,
        math.ceil(float(max_x) / tile_size_m) * tile_size_m,
        math.floor(float(min_y) / tile_size_m) * tile_size_m,
        math.ceil(float(max_y) / tile_size_m) * tile_size_m,
    )


def apply_ml_normalization(arr: np.ndarray, policy: MlScalingPolicy) -> np.ndarray:
    """Return the production ML array with no local statistics."""
    out = np.asarray(arr, dtype="float32")
    if policy.normalization_policy == "preserve_float32":
        return out
    clipped = np.clip(out, float(policy.ml_clip_min), float(policy.ml_clip_max))
    return ((clipped - float(policy.ml_clip_min)) / (float(policy.ml_clip_max) - float(policy.ml_clip_min))).astype("float32")


def render_ml_preview(arr: np.ndarray, policy: MlScalingPolicy) -> np.ndarray:
    """Render an 8-bit PNG preview using fixed constants only."""
    values = np.asarray(arr, dtype="float32")
    finite = np.isfinite(values)
    scaled = np.zeros(values.shape, dtype="float32")
    scaled[finite] = (values[finite] - policy.preview_min) / (policy.preview_max - policy.preview_min)
    return (np.clip(scaled, 0.0, 1.0) * 255.0).round().astype("uint8")


def load_ml_policy(path: str | Path | None = None) -> MlScalingPolicy:
    """Load policy JSON/YAML-ish config if present, otherwise return defaults."""
    if path is None:
        return MlScalingPolicy()
    config_path = Path(path)
    if not config_path.exists():
        return MlScalingPolicy()
    text = config_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload: dict[str, Any] = {}
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            value = value.strip()
            if value.lower() in {"null", "none"}:
                payload[key.strip()] = None
            else:
                try:
                    payload[key.strip()] = float(value)
                except ValueError:
                    payload[key.strip()] = value.strip("'\"")
    return MlScalingPolicy(**payload)


def validate_production_filename(path: str | Path) -> None:
    """Reject output names that imply visual-only enhancement."""
    name = Path(path).name.lower()
    matches = [token for token in FORBIDDEN_PRODUCTION_NAME_TOKENS if token in name]
    if matches:
        raise ValueError(f"Production ML output name contains forbidden token(s): {', '.join(matches)}")
