"""Configuration constants for the lunar tile pipeline."""

from __future__ import annotations

from pathlib import Path

MOON_RADIUS_M = 1_737_400.0
DEFAULT_TILE_SIZE_KM = 10.0
DEFAULT_TILE_PIXELS = 1024
DEFAULT_OUTPUT_DIR = Path("data/tiles")
PROJECTION_NAME = "spherical_equirectangular_moon_first_version"
