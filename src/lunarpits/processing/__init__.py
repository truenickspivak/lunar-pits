"""Data acquisition and geospatial processing utilities."""

from lunarpits.processing.identifiers import normalize_product_id
from lunarpits.processing.lroc_product import (
    LrocProcessingPlan,
    LrocProcessingResult,
    process_product,
    process_product_to_geotiff,
)
from lunarpits.processing.paths import windows_path_to_wsl

__all__ = [
    "LrocProcessingPlan",
    "LrocProcessingResult",
    "RasterQaResult",
    "normalize_product_id",
    "process_product",
    "process_product_to_geotiff",
    "qa_lroc_tif",
    "windows_path_to_wsl",
]


def __getattr__(name: str):
    if name in {"RasterQaResult", "qa_lroc_tif"}:
        from lunarpits.processing.raster_qa import RasterQaResult, qa_lroc_tif

        return {"RasterQaResult": RasterQaResult, "qa_lroc_tif": qa_lroc_tif}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
