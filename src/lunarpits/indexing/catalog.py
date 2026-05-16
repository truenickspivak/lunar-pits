"""Small data structures for product indexes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductRecord:
    """A located remote sensing product and its archive metadata."""

    product_id: str
    source: str
    uri: str
    instrument: str | None = None
    observation_time: str | None = None

