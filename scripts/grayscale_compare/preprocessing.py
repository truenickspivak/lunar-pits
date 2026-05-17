"""Preprocessing methods for grayscale lunar skylight imagery.

The important distinction in this module is local vs. global scaling.
Per-image normalization can make every tile look visually pleasant, but it also
destroys absolute illumination relationships: a dim, shadow-heavy skylight tile
and a bright, sunlit tile can both be stretched to the same 0..1 range. That is
dangerous for lunar skylights because shadow shape and shadow darkness are part
of the signal.

Fixed/global normalization keeps brightness comparable across observations. It
may look flatter on some individual images, but it preserves physically
meaningful differences for a CNN and for human inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


EPS = 1e-8


@dataclass(frozen=True)
class PercentileConstants:
    """Dataset-level percentile constants used by percentile methods."""

    raw_p2: float
    raw_p98: float
    clahe_p2: float
    clahe_p98: float


def to_float32(img: np.ndarray) -> np.ndarray:
    """Convert an image to float32 without changing its numeric scale."""
    return np.asarray(img, dtype=np.float32)


def normalize_clip(img: np.ndarray, low: float, high: float) -> np.ndarray:
    """Clip with fixed constants, then map to 0..1."""
    arr = to_float32(img)
    clipped = np.clip(arr, float(low), float(high))
    return ((clipped - float(low)) / max(float(high) - float(low), EPS)).astype(np.float32)


def method_a_raw_global_scaling(img: np.ndarray) -> np.ndarray:
    """METHOD A: fixed uint8-style clipping.

    This is globally consistent: the same source value maps to the same model
    value everywhere. It can preserve lunar shadow/illumination relationships
    better than per-image stretches.
    """
    return normalize_clip(img, 8.0, 245.0)


def method_b_per_image_minmax(img: np.ndarray) -> np.ndarray:
    """METHOD B: normalize each image independently.

    This is useful as a debug view, but it can erase illumination consistency.
    For example, a heavily shadowed pit and a bright terrain tile can both be
    stretched to full black/white even if their original calibrated brightness
    distributions were very different.
    """
    arr = to_float32(img)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    return ((arr - lo) / max(hi - lo, EPS)).astype(np.float32)


def method_c_percentile_clipping(img: np.ndarray, constants: PercentileConstants) -> np.ndarray:
    """METHOD C: dataset-level p2/p98 clipping."""
    return normalize_clip(img, constants.raw_p2, constants.raw_p98)


def apply_clahe_uint8(img: np.ndarray, *, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE to a grayscale image as uint8.

    CLAHE improves local terrain visibility, but it is not physically neutral:
    it changes local contrast based on neighborhoods. That can help a human see
    subtle crater/rim texture, but it may also alter shadow morphology. Treat it
    as an experiment, not an automatic production choice.
    """
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = method_b_per_image_minmax(arr)
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(arr)


def method_d_clahe_global_scaling(img: np.ndarray) -> np.ndarray:
    """METHOD D: CLAHE followed by Method A fixed scaling."""
    clahe_img = apply_clahe_uint8(img)
    return method_a_raw_global_scaling(clahe_img)


def method_e_clahe_percentile_clipping(img: np.ndarray, constants: PercentileConstants) -> np.ndarray:
    """METHOD E: CLAHE followed by dataset-level percentile clipping."""
    clahe_img = apply_clahe_uint8(img)
    return normalize_clip(clahe_img, constants.clahe_p2, constants.clahe_p98)


def method_f_fixed_shadow_friendly(img: np.ndarray) -> np.ndarray:
    """Extra fixed option: slightly tighter fixed clip for darker previews.

    This is still global/fixed, so it is more consistent than per-image min/max.
    It tends to make low-albedo terrain and pit shadows easier to compare.
    """
    return normalize_clip(img, 0.0, 220.0)


def method_g_gamma_after_global(img: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """Extra experiment: global scaling followed by one fixed gamma value."""
    base = method_a_raw_global_scaling(img)
    return np.power(np.clip(base, 0.0, 1.0), gamma).astype(np.float32)


def build_methods(constants: PercentileConstants) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Return the comparison method registry.

    New methods such as histogram matching, Retinex, denoising, or adaptive
    sharpening can be added here without changing the visualization code.
    """
    return {
        "A_raw_global_8_245": method_a_raw_global_scaling,
        "B_per_image_minmax": method_b_per_image_minmax,
        "C_dataset_p2_p98": lambda img: method_c_percentile_clipping(img, constants),
        "D_clahe_global_8_245": method_d_clahe_global_scaling,
        "E_clahe_dataset_p2_p98": lambda img: method_e_clahe_percentile_clipping(img, constants),
        "F_fixed_0_220": method_f_fixed_shadow_friendly,
        "G_global_gamma_0p8": method_g_gamma_after_global,
    }


def compute_percentile_constants(sample_images: list[np.ndarray]) -> PercentileConstants:
    """Compute reproducible dataset-level percentiles from sampled images."""
    raw_values: list[np.ndarray] = []
    clahe_values: list[np.ndarray] = []
    for img in sample_images:
        arr = to_float32(img)
        raw_values.append(arr[np.isfinite(arr)].ravel())
        clahe = apply_clahe_uint8(img)
        clahe_values.append(clahe.astype(np.float32).ravel())
    raw = np.concatenate([v for v in raw_values if v.size])
    clahe_all = np.concatenate([v for v in clahe_values if v.size])
    raw_p2, raw_p98 = np.percentile(raw, [2.0, 98.0])
    clahe_p2, clahe_p98 = np.percentile(clahe_all, [2.0, 98.0])
    return PercentileConstants(float(raw_p2), float(raw_p98), float(clahe_p2), float(clahe_p98))
