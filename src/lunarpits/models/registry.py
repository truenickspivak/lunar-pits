"""Model registry data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Configuration metadata for a trainable or deployable model."""

    name: str
    task: str
    version: str = "0.1.0"

