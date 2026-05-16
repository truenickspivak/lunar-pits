"""Annotation data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Annotation:
    """A labeled target tied to a source product or tile."""

    annotation_id: str
    label: str
    source_id: str

