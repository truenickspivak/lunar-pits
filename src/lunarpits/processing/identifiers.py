"""Identifier normalization helpers for LROC products."""

from __future__ import annotations


def normalize_product_id(product_id: str) -> str:
    """Return a canonical uppercase LROC product id without file suffixes.

    The function intentionally mirrors the normalization used by the existing
    ISIS controller script while staying pure and import-safe for tests.
    """
    normalized = product_id.strip().upper()
    normalized = normalized.removesuffix(".IMG")
    normalized = normalized.removesuffix(".XML")
    return normalized

