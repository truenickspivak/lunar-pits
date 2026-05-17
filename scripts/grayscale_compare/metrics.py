"""Quantitative metrics for lunar grayscale preprocessing comparisons."""

from __future__ import annotations

import cv2
import numpy as np

try:  # scikit-image is preferred when installed, but OpenCV fallback works.
    from skimage.filters import sobel as skimage_sobel
    from skimage.measure import shannon_entropy
except Exception:  # pragma: no cover - depends on optional local install
    skimage_sobel = None
    shannon_entropy = None


def as_float01(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.max(initial=0) > 1.0 or arr.min(initial=0) < 0.0:
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        arr = (arr - lo) / max(hi - lo, 1e-8)
    return np.clip(arr, 0.0, 1.0)


def entropy(img: np.ndarray) -> float:
    arr = as_float01(img)
    if shannon_entropy is not None:
        return float(shannon_entropy(arr))
    hist, _ = np.histogram(arr, bins=256, range=(0.0, 1.0), density=True)
    probs = hist / max(hist.sum(), 1e-8)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def local_contrast(img: np.ndarray, ksize: int = 9) -> float:
    """Mean local standard deviation, a cheap terrain-detail proxy."""
    arr = as_float01(img)
    mean = cv2.blur(arr, (ksize, ksize))
    mean_sq = cv2.blur(arr * arr, (ksize, ksize))
    local_var = np.maximum(mean_sq - mean * mean, 0.0)
    return float(np.sqrt(local_var).mean())


def edge_density(img: np.ndarray, threshold_percentile: float = 95.0) -> float:
    """Fraction of pixels with strong Sobel/Canny-like gradients."""
    arr = as_float01(img)
    if skimage_sobel is not None:
        grad = skimage_sobel(arr)
    else:
        gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
    cutoff = np.percentile(grad, threshold_percentile)
    return float((grad > cutoff).mean())


def summarize(img: np.ndarray) -> dict[str, float]:
    arr = as_float01(img)
    return {
        "mean_intensity": float(arr.mean()),
        "std_intensity": float(arr.std()),
        "entropy": entropy(arr),
        "local_contrast": local_contrast(arr),
        "edge_density": edge_density(arr),
    }
