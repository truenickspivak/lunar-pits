"""Rasterio QA helpers for processed LROC GeoTIFFs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RasterQaResult:
    """Summary of basic QA checks for a processed raster."""

    tif_path: str
    quicklook_path: str
    driver: str
    width: int
    height: int
    band_count: int
    dtypes: tuple[str, ...]
    crs: str | None
    bounds: tuple[float, float, float, float]
    transform: tuple[float, ...]
    nodata: float | int | None
    valid_pixel_count: int
    finite_pixel_count: int
    nonzero_pixel_count: int
    min_value: float
    max_value: float
    mean_value: float


def default_quicklook_path(tif_path: Path) -> Path:
    return tif_path.with_suffix(".quicklook.png")


def _scaled_quicklook(data):
    import numpy as np

    values = data.compressed()
    if values.size == 0:
        raise ValueError("Raster contains no valid pixels.")

    p2, p98 = np.percentile(values, [2, 98])
    if p98 <= p2:
        p2 = float(values.min())
        p98 = float(values.max())

    if p98 <= p2:
        return np.zeros(data.shape, dtype=np.uint8)

    scaled = np.clip((data.filled(p2) - p2) / (p98 - p2), 0, 1)
    return (scaled * 255).astype(np.uint8)


def qa_lroc_tif(
    tif_path: str | Path,
    *,
    quicklook_path: str | Path | None = None,
    max_size: int = 2048,
) -> RasterQaResult:
    """Open a processed LROC TIFF, validate pixels, and save a PNG quicklook."""
    import numpy as np
    import rasterio
    from matplotlib import pyplot as plt

    tif_path = Path(tif_path)
    quicklook_path = Path(quicklook_path) if quicklook_path else default_quicklook_path(tif_path)

    if max_size <= 0:
        raise ValueError("max_size must be positive.")

    with rasterio.open(tif_path) as src:
        scale = min(max_size / src.width, max_size / src.height, 1.0)
        out_height = max(1, int(src.height * scale))
        out_width = max(1, int(src.width * scale))

        preview = src.read(1, masked=True, out_shape=(out_height, out_width))
        full_band = src.read(1, masked=True)

        valid_values = full_band.compressed()
        if valid_values.size == 0:
            raise ValueError(f"Raster contains no valid pixels: {tif_path}")

        finite_values = valid_values[np.isfinite(valid_values)]
        if finite_values.size == 0:
            raise ValueError(f"Raster contains no finite valid pixels: {tif_path}")

        nonzero_count = int(np.count_nonzero(finite_values))
        if nonzero_count == 0:
            raise ValueError(f"Raster valid pixels are all zero: {tif_path}")

        quicklook = _scaled_quicklook(preview)
        quicklook_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(quicklook_path, quicklook, cmap="gray")

        result = RasterQaResult(
            tif_path=str(tif_path),
            quicklook_path=str(quicklook_path),
            driver=src.driver,
            width=src.width,
            height=src.height,
            band_count=src.count,
            dtypes=tuple(src.dtypes),
            crs=str(src.crs) if src.crs else None,
            bounds=tuple(src.bounds),
            transform=tuple(src.transform)[:6],
            nodata=src.nodata,
            valid_pixel_count=int(valid_values.size),
            finite_pixel_count=int(finite_values.size),
            nonzero_pixel_count=nonzero_count,
            min_value=float(finite_values.min()),
            max_value=float(finite_values.max()),
            mean_value=float(finite_values.mean()),
        )

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QA a processed LROC GeoTIFF and save a quicklook PNG.")
    parser.add_argument("tif_path", help="Path to a processed LROC GeoTIFF")
    parser.add_argument("--quicklook-path", default=None, help="Optional output PNG path")
    parser.add_argument("--max-size", type=int, default=2048, help="Maximum preview width or height in pixels")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result = qa_lroc_tif(
            args.tif_path,
            quicklook_path=args.quicklook_path,
            max_size=args.max_size,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
