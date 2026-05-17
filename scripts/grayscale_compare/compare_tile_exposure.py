"""Compare exposure/scaling policies on float LROC tile TIFFs.

This is intentionally separate from the production gatherer. It reads the same
float TIFFs the model would ingest and writes visual PNG grids so exposure
choices can be judged without accidentally changing the dataset pipeline.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NODATA_DEFAULT = -9999.0


def read_tile(path: Path) -> np.ndarray:
    try:
        import rasterio

        with rasterio.open(path) as src:
            arr = src.read(1).astype("float32")
            nodata = src.nodata
    except Exception:
        import cv2

        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"Could not read {path}")
        arr = np.asarray(arr, dtype="float32")
        nodata = None
    invalid = ~np.isfinite(arr)
    if nodata is not None:
        invalid |= arr == nodata
    invalid |= arr == NODATA_DEFAULT
    out = arr.copy()
    out[invalid] = np.nan
    return out


def valid_values(arrays: list[np.ndarray]) -> np.ndarray:
    values = [arr[np.isfinite(arr)].ravel() for arr in arrays]
    values = [v for v in values if v.size]
    if not values:
        raise ValueError("No finite pixels found.")
    return np.concatenate(values)


def scale_fixed(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((arr - low) / max(high - low, 1e-12), 0.0, 1.0)


def scale_percentile(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    return scale_fixed(arr, low, high)


def scale_asinh(arr: np.ndarray, low: float, high: float, softness: float = 0.18) -> np.ndarray:
    base = scale_fixed(arr, low, high)
    return np.arcsinh(base / softness) / np.arcsinh(1.0 / softness)


def scale_log(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    base = scale_fixed(arr, low, high)
    return np.log1p(9.0 * base) / np.log1p(9.0)


def scale_gamma(arr: np.ndarray, low: float, high: float, gamma: float) -> np.ndarray:
    return np.power(scale_fixed(arr, low, high), gamma)


def scale_soft_percentile(arr: np.ndarray, low_pct: float, high_pct: float, gamma: float = 1.0) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return np.zeros_like(arr, dtype="float32")
    low, high = np.percentile(finite, [low_pct, high_pct])
    scaled = scale_fixed(arr, float(low), float(high))
    return np.power(scaled, gamma).astype("float32")


def scale_median_window(arr: np.ndarray, half_width: float, gamma: float = 1.0) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return np.zeros_like(arr, dtype="float32")
    median = float(np.median(finite))
    scaled = scale_fixed(arr, median - half_width, median + half_width)
    return np.power(scaled, gamma).astype("float32")


def scale_mad_sigmoid(arr: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return np.zeros_like(arr, dtype="float32")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_sigma = max(1.4826 * mad, 1e-6)
    z = (arr - median) / (sigma * robust_sigma)
    return (1.0 / (1.0 + np.exp(-z))).astype("float32")


def scale_mean_std(arr: np.ndarray, width_sigma: float = 2.5, gamma: float = 1.0) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return np.zeros_like(arr, dtype="float32")
    mean = float(np.mean(finite))
    std = max(float(np.std(finite)), 1e-6)
    scaled = scale_fixed(arr, mean - width_sigma * std, mean + width_sigma * std)
    return np.power(scaled, gamma).astype("float32")


def scale_blend_global_local(arr: np.ndarray, global_low: float, global_high: float, local_low_pct: float, local_high_pct: float, local_weight: float) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return np.zeros_like(arr, dtype="float32")
    local_low, local_high = np.percentile(finite, [local_low_pct, local_high_pct])
    low = (1.0 - local_weight) * global_low + local_weight * float(local_low)
    high = (1.0 - local_weight) * global_high + local_weight * float(local_high)
    return scale_fixed(arr, low, high)


def make_methods(arrays: list[np.ndarray]) -> dict[str, callable]:
    values = valid_values(arrays)
    g_p05, g_p1, g_p2, g_p98, g_p99, g_p995 = np.percentile(values, [0.5, 1, 2, 98, 99, 99.5])
    return {
        "fixed_0_0p06": lambda arr: scale_fixed(arr, 0.0, 0.06),
        "fixed_0_0p10_current": lambda arr: scale_fixed(arr, 0.0, 0.10),
        "fixed_0p003_0p08": lambda arr: scale_fixed(arr, 0.003, 0.08),
        "group_p1_p99": lambda arr: scale_percentile(arr, float(g_p1), float(g_p99)),
        "group_p05_p995": lambda arr: scale_percentile(arr, float(g_p05), float(g_p995)),
        "group_p1_p99_gamma0p75": lambda arr: scale_gamma(arr, float(g_p1), float(g_p99), 0.75),
        "group_p1_p99_asinh": lambda arr: scale_asinh(arr, float(g_p1), float(g_p99)),
        "group_p2_p98_log": lambda arr: scale_log(arr, float(g_p2), float(g_p98)),
        "per_tile_p1_p99_DEBUG": lambda arr: scale_percentile(arr, *np.percentile(arr[np.isfinite(arr)], [1, 99])),
    }


def make_adaptive_methods(arrays: list[np.ndarray]) -> dict[str, callable]:
    values = valid_values(arrays)
    g_p1, g_p99 = np.percentile(values, [1, 99])
    return {
        "soft_p0p5_p99p5": lambda arr: scale_soft_percentile(arr, 0.5, 99.5),
        "soft_p1_p99": lambda arr: scale_soft_percentile(arr, 1.0, 99.0),
        "soft_p2_p98": lambda arr: scale_soft_percentile(arr, 2.0, 98.0),
        "soft_p0p5_p99p5_gamma0p85": lambda arr: scale_soft_percentile(arr, 0.5, 99.5, gamma=0.85),
        "soft_p2_p99p7_gamma0p75": lambda arr: scale_soft_percentile(arr, 2.0, 99.7, gamma=0.75),
        "median_window_0p025": lambda arr: scale_median_window(arr, 0.025),
        "median_window_0p035_gamma0p85": lambda arr: scale_median_window(arr, 0.035, gamma=0.85),
        "mean_std_2p5": lambda arr: scale_mean_std(arr, 2.5),
        "mad_sigmoid_3sigma": lambda arr: scale_mad_sigmoid(arr, 3.0),
        "blend_50global_50local": lambda arr: scale_blend_global_local(arr, float(g_p1), float(g_p99), 1.0, 99.0, 0.50),
        "blend_25global_75local": lambda arr: scale_blend_global_local(arr, float(g_p1), float(g_p99), 1.0, 99.0, 0.75),
        "per_tile_p1_p99_reference": lambda arr: scale_soft_percentile(arr, 1.0, 99.0),
    }


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    out = np.clip(out * 255.0, 0, 255).astype("uint8")
    plt.imsave(path, out, cmap="gray", vmin=0, vmax=255)


def write_comparison(paths: list[Path], out_dir: Path, *, adaptive: bool = False) -> None:
    arrays = [read_tile(path) for path in paths]
    methods = make_adaptive_methods(arrays) if adaptive else make_methods(arrays)
    processed_dir = out_dir / "processed"
    rows: list[dict[str, object]] = []

    for path, arr in zip(paths, arrays):
        finite = arr[np.isfinite(arr)]
        for name, fn in methods.items():
            scaled = fn(arr)
            save_png(processed_dir / name / f"{path.stem}.png", scaled)
        rows.append(
            {
                "image": path.name,
                "raw_min": float(np.nanmin(arr)),
                "raw_max": float(np.nanmax(arr)),
                "raw_p1": float(np.nanpercentile(arr, 1)),
                "raw_p50": float(np.nanpercentile(arr, 50)),
                "raw_p99": float(np.nanpercentile(arr, 99)),
                "raw_mean": float(np.nanmean(finite)),
                "raw_std": float(np.nanstd(finite)),
            }
        )

    fig, axes = plt.subplots(len(paths), len(methods), figsize=(2.4 * len(methods), 2.4 * len(paths)), squeeze=False)
    for r, (path, arr) in enumerate(zip(paths, arrays)):
        for c, (name, fn) in enumerate(methods.items()):
            axes[r, c].imshow(fn(arr), cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(name, fontsize=8)
            if c == 0:
                axes[r, c].set_ylabel(path.stem, fontsize=8)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / ("southwest_tranquillitatis_adaptive_exposure_grid.png" if adaptive else "southwest_tranquillitatis_exposure_grid.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for path, arr in zip(paths, arrays):
        values = arr[np.isfinite(arr)].ravel()
        ax.hist(values, bins=160, histtype="step", label=path.stem, density=True)
    ax.set_xlabel("float tile value")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "raw_value_histograms.png", dpi=180)
    plt.close(fig)

    with (out_dir / "raw_tile_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare exposure policies on float LROC tile TIFFs.")
    parser.add_argument("tiles", nargs="+")
    parser.add_argument("--out-dir", default="data/diagnostics/tile_exposure_compare")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive-but-constrained exposure methods.")
    args = parser.parse_args()
    write_comparison([Path(p) for p in args.tiles], Path(args.out_dir), adaptive=args.adaptive)
    print(Path(args.out_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
