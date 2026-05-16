"""Feature data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSet:
    """Named feature collection produced from imagery or annotations."""

    name: str
    source_id: str
    version: str = "0.1.0"

