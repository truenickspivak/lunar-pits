"""Small data models for location context gathering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TargetLocation:
    lat: float
    lon: float
    radius_km: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class ProcessedNacProduct:
    product_id: str
    status: str
    processed_tif: str | None = None
    context_tif: str | None = None
    quicklook: str | None = None
    crop_tif: str | None = None
    crop_quicklook: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LocationGatherResult:
    target: TargetLocation
    out_dir: Path
    products_csv: Path
    products_parquet: Path
    processed: list[ProcessedNacProduct] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

